import os
from typing import Any, Optional

import pandas as pd
import torch
import torch.nn as nn
import torchvision
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from dataset import BigGANImageDataset, BigGANRandomDataset
from modeling import (
    BigGANDiscriminator,
    BigGANDiscriminatorConfig,
    BigGANGenerator,
    BigGANGeneratorConfig,
)

try:
    from apex.optimizers import FusedAdam as Adam
except ModuleNotFoundError:
    from torch.optim import Adam


class BigGANTrainingModule(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        generator_config = BigGANGeneratorConfig(**config.model.generator)
        discriminator_config = BigGANDiscriminatorConfig(**config.model.discriminator)

        self.generator = BigGANGenerator(generator_config)
        self.generator_ema = BigGANGenerator(generator_config)
        self.discriminator = BigGANDiscriminator(discriminator_config)

    def training_step(
        self,
        batch: dict[str, dict[str, torch.Tensor]],
        batch_idx: int,
        optimizer_idx: int,
    ) -> torch.Tensor:
        if optimizer_idx == 0:
            self.generator.requires_grad_(True)
            self.generator.train()
            self.discriminator.requires_grad_(False)
            self.discriminator.eval()

            images = self.generator(**batch["random_g"])
            loss = -self.discriminator(images, batch["random_g"]["labels"]).mean()
            self.log("train/generator_loss", loss)
        elif optimizer_idx == 1:
            self.generator.requires_grad_(False)
            self.generator.eval()
            self.discriminator.requires_grad_(True)
            self.discriminator.train()

            fake_images = self.generator(**batch["random_d"])
            fake_logits = self.discriminator(fake_images, batch["random_d"]["labels"])
            real_logits = self.discriminator(**batch["image"])

            loss = (1 - real_logits).relu().mean() + (1 + fake_logits).relu().mean()
            self.log("train/discriminator_loss", loss)
        return loss

    @torch.no_grad()
    def on_before_optimizer_step(self, optimizer: Optimizer, optimizer_idx: int):
        if optimizer_idx != 0:
            return

        decay = 0.0
        if self.global_step > self.config.optim.generator_ema.start_after:
            decay = self.config.optim.generator_ema.decay

        for p1, p2 in zip(self.generator.parameters(), self.generator_ema.parameters()):
            p2.copy_(decay * p2.float() + (1 - decay) * p1.float())
        for b1, b2 in zip(self.generator.buffers(), self.generator_ema.buffers()):
            p2.copy_(decay * p2.float() + (1 - decay) * p1.float())

    def on_validation_epoch_start(self):
        for module in self.generator_ema.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = None
                module.reset_running_stats()
                module.train()

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self.generator_ema(**batch)

    def validation_epoch_end(self, outputs: list[torch.Tensor]):
        grid = torchvision.utils.make_grid(outputs[-1], value_range=(-1, 1))
        self.logger.log_image("val/images", [grid], self.global_step)

    def configure_optimizers(self) -> tuple[dict[str, Any]]:
        generator_params = self.generator.parameters()
        discriminator_params = self.discriminator.parameters()

        generator_optimizer = {
            "optimizer": Adam(generator_params, **self.config.optim.generator),
            "frequency": 1,
        }
        discriminator_optimizer = {
            "optimizer": Adam(discriminator_params, **self.config.optim.discriminator),
            "frequency": self.config.optim.num_discriminator_steps,
        }
        return generator_optimizer, discriminator_optimizer

    def on_load_checkpoint(self, checkpoint: dict[str, Any]):
        if "amp_scaling_state" in checkpoint:
            checkpoint.pop("amp_scaling_state")


class BigGANDataModule(LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.dataloader_workers = (
            config.dataset.dataloader_workers
            if config.dataset.dataloader_workers > 0
            else os.cpu_count()
        )

    def setup(self, stage: Optional[str] = None):
        dataset = pd.read_json(self.config.dataset.filename, lines=True, dtype=False)
        dataset = dataset[dataset.with_cover]
        dataset = dataset[dataset.cover_aspect_ratio < 0.9]
        self.labels = sorted(dataset.category.unique())

        self.train_image_dataset = BigGANImageDataset(
            dataset,
            image_dir=self.config.dataset.image_dir,
            image_size=self.config.dataset.image_size,
            labels=self.labels,
        )
        self.train_random_dataset = BigGANRandomDataset(
            dataset,
            latent_dim=self.config.model.generator.latent_dim,
            num_labels=len(self.labels),
            num_samples=len(self.train_image_dataset),
            truncation=None,
        )
        self.val_random_dataset = BigGANRandomDataset(
            dataset,
            latent_dim=self.config.model.generator.latent_dim,
            num_labels=len(self.labels),
            num_samples=self.config.optim.num_standing_stats,
            truncation=self.config.model.truncation,
        )

    def train_dataloader(self) -> dict[str, DataLoader]:
        image_dataloader = DataLoader(
            self.train_image_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=True,
            num_workers=self.dataloader_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        random_g_dataloader = DataLoader(
            self.train_random_dataset,
            batch_size=self.config.train.batch_size,
            num_workers=self.dataloader_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        random_d_dataloader = DataLoader(
            self.train_random_dataset,
            batch_size=self.config.train.batch_size,
            num_workers=self.dataloader_workers,
            pin_memory=True,
            persistent_workers=True,
        )
        return {
            "image": image_dataloader,
            "random_g": random_g_dataloader,
            "random_d": random_d_dataloader,
        }

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_random_dataset,
            batch_size=self.config.train.batch_size,
            num_workers=self.dataloader_workers,
            pin_memory=True,
            persistent_workers=True,
        )