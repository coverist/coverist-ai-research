import argparse
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from lightning import VQGANTrainingModule


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
        embedding_dim: int = 256,
        num_layers: tuple[int, ...] = (8, 4, 4, 2, 2),
        hidden_dims: tuple[int, ...] = (512, 256, 256, 128, 128),
    ):
        super().__init__()
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.stem = nn.Conv2d(embedding_dim, hidden_dims[0], kernel_size=1)
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
            self.apply(self.init_weights)
        elif isinstance(module, nn.Conv2d):
            nn.init.orthogonal_(module.weight)
            nn.utils.parametrizations.spectral_norm(module)

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


@torch.no_grad()
def main(args: argparse.Namespace, config: DictConfig):
    model = VQGANTrainingModule.load_from_checkpoint(args.checkpoint, config=config)
    model.cpu().eval()

    # Copy the parameter weights from checkpoint to modified VQGAN model.
    decoder_config = {
        **config.model.decoder,
        "num_embeddings": config.model.quantizer.num_embeddings,
    }
    decoder = VQGANDecoder(**decoder_config)
    decoder.load_state_dict(model.decoder.state_dict(), strict=False)

    # Replace the embeddings of modified VQGAN model with normalized-expanded
    # quantization embedding vectors.
    embedding_weights = F.normalize(model.quantizer.embeddings.weight, eps=1e-6)
    embedding_weights = model.quantizer.expansion(embedding_weights[:, :, None, None])
    decoder.embeddings.weight.copy_(embedding_weights.squeeze())

    torch.save(
        {"state_dict": decoder.state_dict(), "config": decoder_config}, args.output
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--output", default="vqgan-decoder.pth")
    args, unknown_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(unknown_args)
    main(args, config)
