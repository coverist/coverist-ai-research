import math
import os
from typing import Any, Optional

import easyocr
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule
from sklearn.model_selection import train_test_split
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import _LRScheduler as LRScheduler
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid

from dataset import ImageDataset
from modeling import VQVAEDecoder, VQVAEDecoderConfig, VQVAEEncoder, VQVAEEncoderConfig

try:
    from apex.optimizers import FusedAdam as Adam
except ModuleNotFoundError:
    from torch.optim import Adam


class VQVAETrainingModule(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.temperature_start = config.optim.temperature.start
        self.temperature_end = config.optim.temperature.end
        self.temperature_decay_steps = config.optim.temperature.num_decay_steps

        self.encoder = VQVAEEncoder(VQVAEEncoderConfig(**config.model.encoder))
        self.decoder = VQVAEDecoder(VQVAEDecoderConfig(**config.model.decoder))

        self.ocr = easyocr.Reader(["ko"]).detector.module.basenet
        self.ocr.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, ...]:
        recon = self.decoder(self.encoder(images))
        loss_recon = F.l1_loss(images, recon)

        self.ocr.eval()
        loss_ocr_list = [
            F.mse_loss(ocr_images, ocr_recon)
            for ocr_images, ocr_recon in zip(self.ocr(images), self.ocr(recon))
        ]
        loss_ocr = sum(loss_ocr_list) / len(loss_ocr_list)

        loss = loss_recon + loss_ocr
        return recon, loss, loss_recon, loss_ocr

    def training_step(self, images: torch.Tensor, batch_idx: int) -> torch.Tensor:
        _, loss, loss_recon, loss_ocr = self(images)
        self.log("train/loss", loss)
        self.log("train/loss_recon", loss_recon)
        self.log("train/loss_ocr", loss_ocr)
        self.log("train/temperature", self.encoder.temperature)
        self.log("step", self.global_step)
        return loss

    def validation_step(
        self, images: torch.Tensor, batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        recon, loss, loss_recon, loss_ocr = self(images)
        self.log("val/loss", loss)
        self.log("val/loss_recon", loss_recon)
        self.log("val/loss_ocr", loss_ocr)
        self.log("step", self.global_step)
        return images, recon

    def validation_epoch_end(self, outputs: list[tuple[torch.Tensor, torch.Tensor]]):
        images = torch.stack(outputs[0], dim=1).flatten(0, 1)
        images = make_grid(
            images,
            nrow=int(images.size(0) ** 0.5) // 2 * 2,
            value_range=(-1, 1),
        )
        self.logger.log_image("val/reconstructed", [images])

    def configure_optimizers(self) -> tuple[list[Optimizer], list[LRScheduler]]:
        optimizer = Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            **self.config.optim.optimizer
        )
        scheduler = CosineAnnealingLR(optimizer, self.config.train.epochs)
        return [optimizer], [scheduler]

    def on_before_zero_grad(self, optimizer: Optimizer):
        ratio = min(self.global_step / self.temperature_decay_steps, 1.0)
        ratio = 0.5 * (math.cos(math.pi * ratio) + 1)
        self.encoder.temperature = (
            self.temperature_end
            + (self.temperature_start - self.temperature_end) * ratio
        )

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
