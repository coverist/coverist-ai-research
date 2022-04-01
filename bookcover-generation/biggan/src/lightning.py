import os
import random
from typing import Any, Optional

import pandas as pd
import torch
import torch.nn as nn
from dataset import BigGANImageDataset, BigGANRandomDataset
from modeling import (
    BigGANDiscriminator,
    BigGANDiscriminatorConfig,
    BigGANGenerator,
    BigGANGeneratorConfig,
)
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

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
            b2.copy_(decay * b2.float() + (1 - decay) * b1.float())

    def on_validation_epoch_start(self):
        for module in self.generator_ema.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = None
                module.reset_running_stats()
        self.generator_ema.train()

    def validation_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> dict[str, torch.Tensor]:
        batch_size = random.randint(1, batch["latents"].size(0) + 1)
        self.generator_ema(batch["latents"][:batch_size], batch["labels"][:batch_size])
        return batch

    def validation_epoch_end(self, outputs: list[torch.Tensor]):
        self.generator_ema.eval()
        images = self.generator_ema(**outputs[-1])
        self.logger.log_image(
            "val/images",
            [make_grid(images, nrow=int(images.size(0) ** 0.5), value_range=(-1, 1))],
        )

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
        if "ApexMixedPrecisionPlugin" in checkpoint:
            checkpoint.pop("ApexMixedPrecisionPlugin")


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

        num_standing_samples = (
            self.config.optim.num_standing_stats * self.config.train.batch_size
        )
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
            num_samples=num_standing_samples,
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
