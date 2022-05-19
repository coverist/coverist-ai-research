from __future__ import annotations

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


class VQGANDecoder(nn.Module):
    def __init__(
        self,
        num_channels: int = 3,
        num_embeddings: int = 16384,
        num_layers: tuple[int, ...] = (2, 2, 2, 2, 2),
        hidden_dims: tuple[int, ...] = (512, 256, 256, 128, 128),
    ):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings, hidden_dims[0])
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

    def forward(self, latent_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.embeddings(latent_ids).permute(0, 3, 1, 2)
        hidden = self.stem(hidden)
        for i, layers in enumerate(self.blocks):
            for layer in layers:
                hidden = layer(hidden)
            if i < len(self.blocks) - 1:
                hidden = F.interpolate(hidden, scale_factor=2, mode="nearest")
        hidden = self.head(hidden)
        return hidden.tanh()

    @staticmethod
    def from_pretrained(model_path: str) -> VQGANDecoder:
        state_dict = torch.load(model_path)
        model = VQGANDecoder(**state_dict["config"])
        model.load_state_dict(state_dict["state_dict"])
        return model
