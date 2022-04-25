from __future__ import annotations

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
class VQVAEDecoderConfig:
    num_channels: int = 3
    num_layers: tuple[int, ...] = (4, 4, 2, 2, 2)
    hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256, 128)
    middle_reduction: int = 4
    num_embeddings: int = 8192
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


class VQVAEDecoder(nn.Module):
    def __init__(self, config: VQVAEDecoderConfig):
        super().__init__()
        self.embeddings = nn.Embedding(config.num_embeddings, config.embedding_dim)
        self.stem = nn.Conv2d(config.embedding_dim, config.hidden_dims[0], 1)
        self.layers = nn.Sequential(*map(VQVAELayer, config))
        self.head = nn.Conv2d(config.hidden_dims[-1], config.num_channels, 1)

    def forward(self, latent_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embeddings(latent_ids).permute(0, 3, 1, 2)
        hidden = self.stem(hidden)
        hidden = self.layers(hidden)
        hidden = self.head(hidden.relu())
        return hidden.clamp(-1, 1)

    @staticmethod
    def from_pretrained(model_path: str) -> VQVAEDecoder:
        state_dict = torch.load(model_path)
        model = VQVAEDecoder(VQVAEDecoderConfig(**state_dict["config"]))
        model.load_state_dict(state_dict["state_dict"])
        return model
