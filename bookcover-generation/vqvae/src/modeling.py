from collections.abc import Generator
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VQVAEQuantizerConfig:
    num_embeddings: int = 8192
    embedding_dim: int = 128
    temperature: float = 1.0


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


class VQVAEQuantizer(nn.Module):
    def __init__(self, config: VQVAEQuantizerConfig):
        super().__init__()
        self.config = config
        self.temperature = config.temperature
        self.embeddings = nn.Embedding(config.num_embeddings, config.embedding_dim)

    def forward(
        self,
        logits: Optional[torch.Tensor] = None,
        quantized: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if quantized is None and not self.training:
            quantized = logits.argmax(dim=1)
        if quantized is not None:
            return quantized, self.embeddings(quantized).permute(0, 3, 1, 2)

        quantized = F.gumbel_softmax(logits, tau=self.temperature, dim=1)
        embeddings = torch.einsum("bnhw,nd->bdhw", quantized, self.embeddings.weight)
        return quantized, embeddings


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
        self.head = nn.Conv2d(config.hidden_dims[-1], config.num_embeddings, 1)

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

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(embeddings)
        hidden = self.layers(hidden)
        hidden = self.head(hidden.relu())
        return hidden
