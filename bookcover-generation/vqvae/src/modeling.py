from collections.abc import Generator
from dataclasses import dataclass
from typing import Optional

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
    num_layers: tuple[int] = (2, 2, 4, 4, 8)
    hidden_dims: tuple[int] = (128, 256, 512, 1024, 2048)
    middle_reduction: int = 4
    num_embeddings: int = 8192
    embedding_dim: int = 128
    temperature: float = 1.0

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
    num_layers: tuple[int] = (8, 4, 4, 2, 2)
    hidden_dims: tuple[int] = (2048, 1024, 512, 256, 128)
    middle_reduction: int = 4
    num_embeddings: int = 8192
    embedding_dim: int = 128

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


class VQVAELayer(nn.Module):
    def __init__(self, config: VQVAELayerConfig):
        super().__init__()
        self.conv1 = nn.Conv2d(config.input_dim, config.middle_dim, 1)
        self.conv2 = nn.Conv2d(config.middle_dim, config.middle_dim, 3, padding=1)
        self.conv3 = nn.Conv2d(config.middle_dim, config.output_dim, 1)

        self.shortcut = (
            nn.Conv2d(config.input_dim, config.output_dim, 1)
            if config.input_dim != config.output_dim
            else nn.Identity()
        )
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
        shortcut = self.upsample(self.shortcut(self.pool(hidden)))
        hidden = self.conv1(hidden.relu())
        hidden = self.conv2(self.upsample(hidden.relu()))
        hidden = self.conv3(self.pool(hidden.relu()))
        return hidden + shortcut


class VQVAEEncoder(nn.Module):
    def __init__(self, config: VQVAEEncoderConfig):
        super().__init__()
        self.stem = nn.Conv2d(config.num_channels, config.hidden_dims[0], 7, padding=3)
        self.layers = nn.Sequential(*map(VQVAELayer, config))
        self.head = nn.Conv2d(config.hidden_dims[-1], config.embedding_dim, 1)

        self.embeddings = nn.Conv2d(config.embedding_dim, config.num_embeddings, 1)
        self.temperature = config.temperature

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(images)
        hidden = self.layers(hidden)
        hidden = self.head(hidden.relu())
        logits = self.embeddings(hidden.relu())

        if self.training:
            return F.gumbel_softmax(logits, self.temperature, eps=1e-6, dim=1)
        return torch.zeros_like(logits).scatter_(1, logits.argmax(1, keepdim=True), 1)


class VQVAEDecoder(nn.Module):
    def __init__(self, config: VQVAEDecoderConfig):
        super().__init__()
        self.embeddings = nn.Conv2d(config.num_embeddings, config.embedding_dim, 1)
        nn.init.normal_(self.embeddings.weight)

        self.stem = nn.Conv2d(config.embedding_dim, config.hidden_dims[0], 1)
        self.layers = nn.Sequential(*map(VQVAELayer, config))
        self.head = nn.Conv2d(config.hidden_dims[-1], config.num_channels, 1)

    def forward(self, encodings: torch.Tensor) -> torch.Tensor:
        hidden = self.embeddings(encodings)
        hidden = self.stem(hidden.relu())
        hidden = self.layers(hidden)
        hidden = self.head(hidden.relu())
        return hidden.tanh()
