from collections.abc import Generator
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


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
    embedding_dim: int = 128

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


@dataclass
class VQVAEQuantizerConfig:
    num_embeddings: int = 8192
    embedding_dim: int = 128
    ema_decay: float = 0.99


class VQVAELayer(nn.Module):
    def __init__(self, config: VQVAELayerConfig):
        super().__init__()
        self.conv1 = nn.Conv2d(config.input_dim, config.middle_dim, 1)
        self.conv2 = nn.Conv2d(config.middle_dim, config.middle_dim, 3, padding=1)
        self.a = nn.Conv2d(config.middle_dim, config.middle_dim, 3, padding=1)
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
            nn.Upsample(None, config.upsampling)
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

    def forward(self, encodings: torch.Tensor) -> torch.Tensor:
        hidden = self.stem(encodings)
        hidden = self.layers(hidden)
        hidden = self.head(hidden.relu())
        return hidden


class VQVAEQuantizer(nn.Module):
    def __init__(self, config: VQVAEQuantizerConfig):
        super().__init__()
        self.config = config
        self.embeddings = nn.Embedding(config.num_embeddings, config.embedding_dim)

        self.embeddings.requires_grad_(False)
        self.embeddings.weight.normal_(0, 0.02)

    def forward(
        self,
        encodings: Optional[torch.Tensor] = None,
        quantized_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if quantized_ids is not None:
            return quantized_ids, self.embeddings(quantized_ids).permute(0, 3, 1, 2)

        batch_size, _, height, width = encodings.shape
        encodings = encodings.permute(0, 2, 3, 1).flatten(0, 2)

        encodings_norm = encodings.square().sum(1).unsqueeze(1)
        embeddings_norm = self.embeddings.weight.square().sum(1).unsqueeze(0)
        dot_product = torch.matmul(encodings, self.embeddings.weight.transpose(0, 1))

        quantized_ids = (encodings_norm + embeddings_norm - 2 * dot_product).argmin(1)
        quantized_encodings = (self.embeddings(quantized_ids) - encodings).detach()
        quantized_encodings = quantized_encodings + encodings

        if self.training:
            num_selected = encodings.new_zeros(self.embeddings.num_embeddings)
            num_selected.scatter_(0, quantized_ids, 1, reduce="add")

            new_embeddings = torch.zeros_like(self.embeddings.weight)
            new_embeddings.scatter_(0, quantized_ids[:, None], encodings, reduce="add")
            new_embeddings = new_embeddings / (num_selected[:, None] + 1e-12)

            self.embeddings.weight.mul_(self.config.ema_decay)
            self.embeddings.weight.add_(new_embeddings, alpha=1 - self.config.ema_decay)

        return (
            quantized_ids.view(batch_size, height, width),
            quantized_encodings.view(batch_size, height, width, -1).permute(0, 3, 1, 2),
        )
