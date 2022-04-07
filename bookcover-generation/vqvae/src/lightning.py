import os
from typing import Any, Optional

import torch
import torch.nn.functional as F
import wandb
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule
from sklearn.model_selection import train_test_split
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid

from dataset import ImageDataset
from modeling import (
    VQVAEDecoder,
    VQVAEDecoderConfig,
    VQVAEEncoder,
    VQVAEEncoderConfig,
    VQVAEQuantizer,
    VQVAEQuantizerConfig,
)

try:
    from apex.optimizers import FusedAdam as Adam
except ModuleNotFoundError:
    from torch.optim import Adam


class VQVAETrainingModule(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.encoder = VQVAEEncoder(VQVAEEncoderConfig(**config.model.encoder))
        self.decoder = VQVAEDecoder(VQVAEDecoderConfig(**config.model.decoder))
        self.quantizer = VQVAEQuantizer(VQVAEQuantizerConfig(**config.model.quantizer))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        logits = self.encoder(images)
        quantized, embeddings = self.quantizer(logits)
        decoded = self.decoder(embeddings)

        log_probs = logits.permute(0, 2, 3, 1).flatten(0, 2).log_softmax(-1)
        log_uniform = (torch.ones_like(log_probs) / logits.size(-1)).log()

        loss_recon = (images - decoded).abs().mean()
        loss_kld = F.kl_div(log_uniform, log_probs, reduce="batchmean", log_target=True)
        loss = loss_recon + 1e-4 * loss_kld
        return quantized, decoded, loss, loss_recon, loss_kld

    def training_step(self, images: torch.Tensor, batch_idx: int) -> torch.Tensor:
        _, _, loss, loss, loss_recon, loss_kld = self(images)
        self.log("train/loss", loss)
        self.log("train/loss_recon", loss_recon)
        self.log("train/loss_kld", loss_kld)
        self.log("step", self.global_step)
        return loss

    def validation_step(
        self, images: torch.Tensor, batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        quantized, decoded, loss, loss_recon, loss_kld = self(images)
        self.log("val/loss", loss)
        self.log("val/loss_recon", loss_recon)
        self.log("val/loss_kld", loss_kld)
        return images, quantized, decoded

    def validation_epoch_end(self, outputs: list[tuple[torch.Tensor, torch.Tensor]]):
        quantized = torch.cat([x[1] for x in outputs], dim=0)
        wandb.log({"val/quantized", wandb.Histogram(quantized.cpu())})

        images = torch.stack((outputs[0][0], outputs[0][2]), dim=1).flatten(0, 1)
        images = make_grid(images, int(images.size(0) ** 0.5), value_range=(-1, 1))
        self.logger.log_image("val/reconstruct", [images])

    def configure_optimizers(self) -> tuple[list[Optimizer], list[LRScheduler]]:
        optimizer = Adam(self.parameters(), **self.config.optim)
        scheduler = CosineAnnealingLR(optimizer, self.config.train.epochs)
        return [optimizer], [scheduler]

    def on_load_checkpoint(self, checkpoint: dict[str, Any]):
        if "ApexMixedPrecisionPlugin" in checkpoint:
            checkpoint.pop("ApexMixedPrecisionPlugin")


class VQVAEDataModule(LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.dataloader_workers = (
            config.dataset.dataloader_workers
            if config.dataset.dataloader_workers > 0
            else os.cpu_count()
        )

    def setup(self, stage: Optional[str] = None):
        dataset = ImageDataset(
            pattern=self.config.dataset.pattern,
            resolution=self.config.dataset.resolution,
        )
        train_indices, val_indices = train_test_split(
            range(len(dataset)),
            test_size=self.config.dataset.validation_ratio,
            random_state=42,
            shuffle=True,
        )
        self.train_dataset = Subset(dataset, train_indices)
        self.val_dataset = Subset(dataset, val_indices)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=True,
            num_workers=self.dataloader_workers,
            pin_memory=True,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.train.batch_size,
            num_workers=self.dataloader_workers,
            pin_memory=True,
            persistent_workers=True,
        )
