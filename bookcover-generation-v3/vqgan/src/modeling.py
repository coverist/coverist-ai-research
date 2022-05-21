import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class VQGANEncoder(nn.Module):
    def __init__(
        self,
        num_channels: int = 3,
        hidden_dims: tuple[int, ...] = (512, 512, 512, 512),
    ):
        super().__init__()
        hidden_dims = (num_channels,) + tuple(hidden_dims)
        self.layers = nn.ModuleList(
            nn.Conv2d(input_dim, output_dim, 4, stride=2, padding=1)
            for input_dim, output_dim in zip(hidden_dims[:-1], hidden_dims[1:])
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            images = layer(images).relu()
        return images


class VQGANDecoder(nn.Module):
    def __init__(
        self,
        num_channels: int = 3,
        hidden_dims: tuple[int, ...] = (512, 512, 512, 512),
    ):
        super().__init__()
        hidden_dims = tuple(hidden_dims) + (num_channels,)
        self.layers = nn.ModuleList(
            nn.ConvTranspose2d(input_dim, output_dim, 4, stride=2, padding=1)
            for input_dim, output_dim in zip(hidden_dims[:-1], hidden_dims[1:])
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            images = layer(images.relu())
        return images.tanh()


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
