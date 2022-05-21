from typing import Optional

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class VQGANLayer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1)

        if input_dim != output_dim:
            self.shortcut = nn.Conv2d(input_dim, output_dim, kernel_size=1)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        shortcut = self.shortcut(hidden) if hasattr(self, "shortcut") else hidden
        hidden = self.conv1(hidden.relu())
        hidden = self.conv2(hidden.relu())
        return hidden + shortcut


class VQGANEncoder(nn.Module):
    def __init__(
        self,
        num_channels: int = 3,
        num_layers: tuple[int, ...] = (2, 2, 2, 2, 2),
        hidden_dims: tuple[int, ...] = (128, 128, 256, 256, 512),
    ):
        super().__init__()
        self.stem = nn.Conv2d(num_channels, hidden_dims[0], kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            nn.ModuleList(
                VQGANLayer(input_dim if i == 0 else output_dim, output_dim)
                for i in range(num_repeats)
            )
            for input_dim, output_dim, num_repeats in zip(
                [hidden_dims[0]] + hidden_dims[:-1], hidden_dims, num_layers
            )
        )
        self.head = nn.Conv2d(hidden_dims[-1], hidden_dims[-1], kernel_size=1)
        self.init_weights()

    @torch.no_grad()
    def init_weights(self, module: Optional[nn.Module] = None):
        if module is None:
            self.stem.apply(self.init_weights)
            self.blocks.apply(self.init_weights)
        elif isinstance(module, nn.Conv2d):
            nn.init.orthogonal_(module.weight)
            nn.utils.parametrizations.spectral_norm(module)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(images)
        for i, layers in enumerate(self.blocks):
            for layer in layers:
                hidden = layer(hidden)
            if i < len(self.blocks) - 1:
                hidden = F.avg_pool2d(hidden, kernel_size=2, stride=2)
        hidden = self.head(hidden)
        return hidden


class VQGANDecoder(nn.Module):
    def __init__(
        self,
        num_channels: int = 3,
        num_layers: tuple[int, ...] = (2, 2, 2, 2, 2),
        hidden_dims: tuple[int, ...] = (512, 256, 256, 128, 128),
    ):
        super().__init__()
        self.stem = nn.Conv2d(hidden_dims[0], hidden_dims[0], kernel_size=1)
        self.blocks = nn.ModuleList(
            nn.ModuleList(
                VQGANLayer(input_dim if i == 0 else output_dim, output_dim)
                for i in range(num_repeats)
            )
            for input_dim, output_dim, num_repeats in zip(
                [hidden_dims[0]] + hidden_dims[:-1], hidden_dims, num_layers
            )
        )
        self.head = nn.Conv2d(hidden_dims[-1], num_channels, kernel_size=3, padding=1)
        self.init_weights()

    @torch.no_grad()
    def init_weights(self, module: Optional[nn.Module] = None):
        if module is None:
            self.stem.apply(self.init_weights)
            self.blocks.apply(self.init_weights)
        elif isinstance(module, nn.Conv2d):
            nn.init.orthogonal_(module.weight)
            nn.utils.parametrizations.spectral_norm(module)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(latents)
        for i, layers in enumerate(self.blocks):
            for layer in layers:
                hidden = layer(hidden)
            if i < len(self.blocks) - 1:
                hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
        hidden = self.head(hidden)
        return hidden.tanh()


class VQGANQuantizer(nn.Module):
    def __init__(
        self,
        num_embeddings: int = 16384,
        embedding_dim: int = 512,
        factorized_dim: int = 32,
    ):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings, factorized_dim)
        self.projection = nn.Conv2d(embedding_dim, factorized_dim, kernel_size=1)
        self.expansion = nn.Conv2d(factorized_dim, embedding_dim, kernel_size=1)

    def forward(
        self, encodings: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encodings = F.normalize(self.projection(encodings), eps=1e-6)
        embeddings = F.normalize(self.embeddings.weight, eps=1e-6)
        cosine_similarities = torch.einsum("bdhw,nd->bnhw", encodings, embeddings)

        closest_indices = cosine_similarities.argmax(dim=1)
        flatten_indices = closest_indices.flatten()

        # Get closest codebook embedding vectors, compute the quantization loss and
        # apply a gradient trick with expanding to the original embedding space.
        latents = F.embedding(closest_indices, embeddings).permute(0, 3, 1, 2)
        loss_quantization = F.mse_loss(encodings, latents)

        latents = encodings + (latents - encodings).detach()
        latents = self.expansion(latents)

        # Calculate the perplexity of quantizations to visualize the codebook usage.
        embedding_usages = flatten_indices.new_zeros(self.embeddings.num_embeddings)
        embedding_usages.scatter_(0, flatten_indices, 1, reduce="add")
        embedding_usages = embedding_usages / flatten_indices.size(0)

        perplexity = -embedding_usages * (embedding_usages + 1e-6).log()
        perplexity = perplexity.sum().exp()

        return latents, closest_indices, loss_quantization, perplexity


class PatchDiscriminator(nn.Sequential):
    def __init__(self, num_channels: int = 3, base_dim: int = 64):
        super().__init__(
            nn.Conv2d(num_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_dim, 2 * base_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(2 * base_dim, 4 * base_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(4 * base_dim, 8 * base_dim, kernel_size=4, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(8 * base_dim, 1, kernel_size=4, padding=1),
        )
        self.init_weights()

    @torch.no_grad()
    def init_weights(self, module: Optional[nn.Module] = None):
        if module is None:
            self.apply(self.init_weights)
        elif isinstance(module, nn.Conv2d):
            nn.init.orthogonal_(module.weight)
            nn.utils.parametrizations.spectral_norm(module)


class PerceptualLoss(nn.Module):
    def __init__(self, architecture: str, pretrained: str):
        super().__init__()
        self.model = timm.create_model(architecture, features_only=True)
        self.model.load_state_dict(torch.load(pretrained), strict=False)
        self.model.requires_grad_(False)

    def forward(self, images: torch.Tensor, decodings: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        features_images, features_decodings = self.model(images), self.model(decodings)
        return sum(map(F.l1_loss, features_images, features_decodings))
