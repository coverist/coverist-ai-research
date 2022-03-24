import os
from typing import Any, Optional

import albumentations as A
import albumentations.pytorch as AP
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.data import DataLoader

from dataset import BigGANDataset
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

        self.noise_size = generator_config.noise_size
        self.num_classes = generator_config.num_classes

        self.generator = BigGANGenerator(generator_config)
        self.generator_ema = BigGANGenerator(generator_config)
        self.discriminator = BigGANDiscriminator(discriminator_config)

        logging_noise_vectors, logging_labels = self.sample_random_inputs(
            config.train.logging_images, truncated=True
        )
        self.register_buffer("logging_noise_vectors", logging_noise_vectors)
        self.register_buffer("logging_labels", logging_labels)

    def sample_random_inputs(
        self, batch_size: Optional[int] = None, truncated: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = batch_size or self.config.train.batch_size
        noise_vector = torch.randn((batch_size, self.noise_size), device=self.device)
        labels = torch.randint(0, self.num_classes, (batch_size,), device=self.device)

        if truncated:
            noise_vector = noise_vector.fmod(self.config.model.truncation_threshold)
        return noise_vector, labels

    def generator_step(self) -> torch.Tensor:
        noise_vectors, labels = self.sample_random_inputs()
        loss = -self.discriminator(self.generator(noise_vectors, labels), labels).mean()
        self.log("train/generator_loss", loss)
        return loss

    def discriminator_step(
        self, real_images: torch.Tensor, real_labels: torch.Tensor
    ) -> torch.Tensor:
        noise_vectors, fake_labels = self.sample_random_inputs(real_images.size(0))
        fake_images = self.generator(noise_vectors, fake_labels)

        real_logits = self.discriminator(real_images, real_labels)
        fake_logits = self.discriminator(fake_images, fake_labels)

        loss = (1 - real_logits).relu().mean() + (1 + fake_logits).relu().mean()
        self.log("train/discriminator_loss", loss)
        return loss

    def training_step(
        self, batch: tuple[torch.Tensor, ...], batch_idx: int, optimizer_idx: int
    ) -> torch.Tensor:
        self.generator.requires_grad_(int(optimizer_idx) == 0)
        self.discriminator.requires_grad_(int(optimizer_idx) == 1)

        if optimizer_idx == 0:
            return self.generator_step()
        elif optimizer_idx == 1:
            return self.discriminator_step(*batch)

    @torch.no_grad()
    def on_before_zero_grad(self, outputs: torch.Tensor):
        decay = 0.0
        if self.global_step > self.config.optim.generator_ema.start_after:
            decay = self.config.optim.generator_ema.decay

        for p1, p2 in zip(self.generator_ema.parameters(), self.generator.parameters()):
            p1.copy_(decay * p1.float() + (1 - decay) * p2.float())
        for p1, p2 in zip(self.generator_ema.buffers(), self.generator.buffers()):
            p1.copy_(decay * p1.float() + (1 - decay) * p2.float())

    @torch.no_grad()
    def training_epoch_end(self, outputs: list[list[torch.Tensor]]):
        if (self.current_epoch + 1) % self.config.train.logging_interval != 0:
            return

        for module in self.generator_ema.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.momentum = None
                module.reset_running_stats()
                module.train()

        for _ in range(self.config.train.accumulate_standing_stats):
            self.generator_ema(*self.sample_random_inputs(truncated=True))
        self.generator_ema.eval()

        images = self.generator_ema(self.logging_noise_vectors, self.logging_labels)
        grid = torchvision.utils.make_grid(images, value_range=(-1, 1))
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

    def setup(self, stage: Optional[str] = None):
        dataset = pd.read_csv(self.config.dataset.filename, dtype={"barcode": "str"})
        dataset = dataset[~dataset.category.isnull()]

        transforms = [
            A.Resize(
                self.config.dataset.image_resolution,
                self.config.dataset.image_resolution,
            ),
            A.Normalize(mean=0.5, std=0.5),
            AP.ToTensorV2(),
        ]

        self.label2id = {v: k for k, v in enumerate(sorted(dataset.category.unique()))}
        self.train_dataset = BigGANDataset(
            dataset,
            self.config.dataset.imagedir,
            self.label2id,
            transform=A.Compose(transforms),
        )

    def train_dataloader(self) -> DataLoader:
        num_workers = (
            self.config.dataset.dataloader_workers
            if self.config.dataset.dataloader_workers >= 0
            else os.cpu_count()
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
        )
