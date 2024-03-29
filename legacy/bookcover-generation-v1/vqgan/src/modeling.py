from collections.abc import Generator
from dataclasses import dataclass
from typing import Any, Optional

import easyocr
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VQVAELayerConfig:
    input_dim: int
    middle_dim: int
    output_dim: int
    pooling: Optional[int] = None
    upsampling: Optional[int] = None


@dataclass
class VQVAEEncoderConfig:
    num_channels: int = 3
    num_layers: tuple[int, ...] = (2, 2, 2, 4, 4)
    hidden_dims: tuple[int, ...] = (128, 256, 512, 1024, 2048)
    middle_reduction: int = 4
    embedding_dim: int = 256

    def __iter__(self) -> Generator[VQVAELayerConfig]:
        for i, num_layers in enumerate(self.num_layers):
            for j in range(num_layers):
                handover = i > 0 and j == 0
                yield VQVAELayerConfig(
                    self.hidden_dims[i - 1 if handover else i],
                    self.hidden_dims[i] // self.middle_reduction,
                    self.hidden_dims[i],
                    pooling=2 if handover else None,
                )


@dataclass
class VQVAEDecoderConfig:
    num_channels: int = 3
    num_layers: tuple[int, ...] = (4, 4, 2, 2, 2)
    hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256, 128)
    middle_reduction: int = 4
    embedding_dim: int = 256

    def __iter__(self) -> Generator[VQVAELayerConfig]:
        for i, num_layers in enumerate(self.num_layers):
            for j in range(num_layers):
                handover = i < len(self.num_layers) - 1 and j == num_layers - 1
                yield VQVAELayerConfig(
                    self.hidden_dims[i],
                    self.hidden_dims[i] // self.middle_reduction,
                    self.hidden_dims[i + 1 if handover else i],
                    upsampling=2 if handover else None,
                )


@dataclass
class VQVAEQuantizerConfig:
    num_embeddings: int = 8192
    embedding_dim: int = 256
    factorized_dim: int = 16


@dataclass
class PatchDiscriminatorConfig:
    num_channels: int = 3
    kernel_size: int = 4
    hidden_dims: tuple[int, ...] = (64, 128, 256)
    num_head_layers: int = 2

    def __iter__(self) -> Generator[tuple[Any, ...]]:
        hidden_dims = [self.num_channels] + list(self.hidden_dims)
        stride_padding = (self.kernel_size - 1) // 2

        for input_dim, output_dim in zip(hidden_dims[:-1], hidden_dims[1:]):
            yield (input_dim, output_dim, self.kernel_size, 2, stride_padding)
        for _ in range(self.num_head_layers):
            yield (output_dim, output_dim, self.kernel_size, 1, "same")


class VQVAELayer(nn.Module):
    def __init__(self, config: VQVAELayerConfig):
        super().__init__()
        self.conv1 = nn.Conv2d(config.input_dim, config.middle_dim, 1)
        self.conv2 = nn.Conv2d(config.middle_dim, config.middle_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(config.middle_dim, config.output_dim, 1)

        self.pool = (
            nn.AvgPool2d(config.pooling)
            if config.pooling is not None
            else nn.Identity()
        )
        self.upsample = (
            nn.Upsample(scale_factor=config.upsampling, mode="nearest")
            if config.upsampling is not None
            else nn.Identity()
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        shortcut = self.upsample(self.pool(hidden))

        hidden = self.conv1(hidden.relu())
        hidden = self.conv2(self.upsample(hidden.relu()))
        hidden = self.conv3(self.pool(hidden.relu()))

        padding_dim = hidden.size(1) - shortcut.size(1)
        if padding_dim < 0:
            shortcut = shortcut[:, :-padding_dim]
        elif shortcut.size(1) < hidden.size(1):
            shortcut = F.pad(shortcut, (0, 0, 0, 0, 0, padding_dim))
        return hidden + shortcut


class VQVAEEncoder(nn.Module):
    def __init__(self, config: VQVAEEncoderConfig):
        super().__init__()
        self.stem = nn.Conv2d(config.num_channels, config.hidden_dims[0], 7, padding=3)
        self.layers = nn.Sequential(*map(VQVAELayer, config))
        self.head = nn.Conv2d(config.hidden_dims[-1], config.embedding_dim, 1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(images)
        hidden = self.layers(hidden)
        hidden = self.head(hidden.relu())
        return hidden


class VQVAEDecoder(nn.Module):
    def __init__(self, config: VQVAEDecoderConfig):
        super().__init__()
        self.stem = nn.Conv2d(config.embedding_dim, config.hidden_dims[0], 1)
        self.layers = nn.Sequential(*map(VQVAELayer, config))
        self.head = nn.Conv2d(config.hidden_dims[-1], config.num_channels, 1)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(latents)
        hidden = self.layers(hidden)
        hidden = self.head(hidden.relu())
        return hidden


class VQVAEQuantizer(nn.Module):
    def __init__(self, config: VQVAEQuantizerConfig):
        super().__init__()
        self.embeddings = nn.Embedding(config.num_embeddings, config.factorized_dim)
        self.projection = nn.Conv2d(config.embedding_dim, config.factorized_dim, 1)
        self.expansion = nn.Conv2d(config.factorized_dim, config.embedding_dim, 1)

    def forward(
        self, encoded: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded_unnormalized = self.projection(encoded)

        encoded = F.normalize(encoded_unnormalized, eps=1e-6)
        embeddings = F.normalize(self.embeddings.weight, eps=1e-6)
        cosine_similarities = torch.einsum("bdhw,nd->bnhw", encoded, embeddings)

        closest_indices = cosine_similarities.argmax(dim=1)
        flatten_indices = closest_indices.flatten()

        latents = F.embedding(closest_indices, embeddings).permute(0, 3, 1, 2)
        loss_quantization = F.mse_loss(encoded_unnormalized, latents)

        latents = encoded + (latents - encoded).detach()
        latents = self.expansion(latents)

        embedding_usages = flatten_indices.new_zeros(self.embeddings.num_embeddings)
        embedding_usages.scatter_(0, flatten_indices, 1, reduce="add")
        embedding_usages = embedding_usages / flatten_indices.size(0)

        perplexity = -embedding_usages * (embedding_usages + 1e-6).log()
        perplexity = perplexity.sum().exp()

        return latents, closest_indices, loss_quantization, perplexity


class PatchDiscriminator(nn.Module):
    def __init__(self, config: PatchDiscriminatorConfig):
        super().__init__()
        self.layers = nn.ModuleList(nn.Conv2d(*args) for args in config)
        self.projection = nn.Conv2d(config.hidden_dims[-1], 1, 1)
        self.init_weights()

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight)
                nn.utils.parametrizations.spectral_norm(module, eps=1e-6)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            hidden = F.leaky_relu(layer(hidden), 0.02)
        return self.projection(hidden)


class OCRPerceptualLoss(nn.Module):
    def __init__(self, input_size: tuple[int, int] = (128, 128)):
        super().__init__()
        self.input_size = input_size

        self.model = easyocr.Reader(["ko"]).detector.module.basenet
        self.model.requires_grad_(False)

        self.register_buffer("shift", torch.tensor([[[[0.03]], [[0.088]], [[0.188]]]]))
        self.register_buffer("scale", torch.tensor([[[[0.458]], [[0.448]], [[0.45]]]]))

    def forward_features(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        images = (images + self.shift) / self.scale
        images = F.interpolate(images, self.input_size, mode="bilinear")

        features = self.model.eval()(images)
        features = [F.normalize(feature, dim=1, eps=1e-6) for feature in features]
        return features

    def forward(self, images: torch.Tensor, decoded: torch.Tensor) -> torch.Tensor:
        images_features = self.forward_features(images)
        decoded_features = self.forward_features(decoded)

        loss = 0
        for images_feature, decoded_feature in zip(images_features, decoded_features):
            loss = loss + (images_feature - decoded_feature).square().sum(1).mean()
        return loss
