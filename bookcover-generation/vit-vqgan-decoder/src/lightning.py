import math
from typing import Any, Optional

import torch
import torch.nn.functional as F
from modeling import Discriminator, OCRPerceptualLoss, PatchToImage
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torchvision.utils import make_grid
from transformers import BertConfig, BertForTokenClassification, get_scheduler

try:
    from apex.optimizers import FusedAdam as AdamW
except ModuleNotFoundError:
    from torch.optim import AdamW


class VQGANTrainingModule(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.num_log_batches = math.ceil(64 / self.config.train.batch_size)

        self.l2_recon_loss_weight = config.optim.criterion.l2_recon
        self.l1_recon_loss_weight = config.optim.criterion.l1_recon
        self.perceptual_loss_weight = config.optim.criterion.perceptual
        self.adversarial_loss_weight = config.optim.criterion.adversarial

        self.decoder = BertForTokenClassification(BertConfig(**config.model.decoder))
        self.patch_to_image = PatchToImage(config.model.discriminator.num_channels)
        self.discriminator = Discriminator(**config.model.discriminator)
        self.perceptual = OCRPerceptualLoss(**config.model.perceptual)

    def generator_step(
        self, latent_ids: torch.Tensor, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        reconstructed = self.decoder(input_ids=latent_ids)[0]
        reconstructed = self.patch_to_image(reconstructed)

        loss_l2_recon = F.mse_loss(images, reconstructed)
        loss_l1_recon = F.l1_loss(images, reconstructed)
        loss_perceptual = self.perceptual(images, reconstructed)
        loss_generator = -self.discriminator(reconstructed).mean()

        loss = (
            self.l2_recon_loss_weight * loss_l2_recon
            + self.l1_recon_loss_weight * loss_l1_recon
            + self.perceptual_loss_weight * loss_perceptual
            + self.adversarial_loss_weight * loss_generator
        )
        metrics = {
            "loss_l2_recon": loss_l2_recon,
            "loss_l1_recon": loss_l1_recon,
            "loss_perceptual": loss_perceptual,
            "loss_generator": loss_generator,
        }
        return reconstructed, loss, metrics

    def discriminator_step(
        self, latent_ids: torch.Tensor, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        reconstructed = self.decoder(input_ids=latent_ids)[0]
        reconstructed = self.patch_to_image(reconstructed)

        loss_discriminator_real = (1 - self.discriminator(images)).relu().mean()
        loss_discriminator_fake = (1 + self.discriminator(reconstructed)).relu().mean()
        loss_discriminator = (loss_discriminator_real + loss_discriminator_fake) / 2

        metrics = {
            "loss_discriminator": loss_discriminator,
            "loss_discriminator_real": loss_discriminator_real,
            "loss_discriminator_fake": loss_discriminator_fake,
        }
        return reconstructed, loss_discriminator, metrics

    def forward(
        self, latent_ids: torch.Tensor, images: torch.Tensor, optimizer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self.training:
            self.decoder.requires_grad_(optimizer_idx == 0)
            self.decoder.train(optimizer_idx == 0)

            self.discriminator.requires_grad_(optimizer_idx == 1)
            self.discriminator.train(optimizer_idx == 1)

        if optimizer_idx == 0:
            return self.generator_step(latent_ids, images)
        elif optimizer_idx == 1:
            return self.discriminator_step(latent_ids, images)

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
        optimizer_idx: int,
    ) -> torch.Tensor:
        _, loss, metrics = self(*batch, int(optimizer_idx))
        self.log("step", self.global_step)
        self.log_dict({f"train/{k}": v for k, v in metrics.items()})
        return loss

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        reconstructed, _, metrics = self(*batch, optimizer_idx=0)
        self.log("step", self.global_step)
        self.log_dict({f"val/{k}": v for k, v in metrics.items()})

        # Prevent from storing unnecessary image tensors which consume large portion of
        # GPU memory and occur OOM at validation.
        if batch_idx < self.num_log_batches:
            return batch[1], reconstructed
        return None, None

    def validation_epoch_end(
        self, outputs: list[tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]
    ):
        if not outputs:
            return

        # Get 64 original and reconstructed images.
        outputs = [output for output in outputs if output[0] is not None]
        images = torch.cat([output[0] for output in outputs])[:64]
        reconstructed = torch.cat([output[1] for output in outputs])[:64]

        # Stack the original and reconstructed images, make a grid, and log the image.
        grid = torch.stack((images, reconstructed), dim=1).flatten(0, 1)
        grid = make_grid(grid, value_range=(-1, 1))
        self.logger.log_image("val/reconstructed", [grid])

    def configure_optimizers(self) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        generator_optimizer = AdamW(
            self.decoder.parameters(), **self.config.optim.optimizer
        )
        discriminator_optimizer = AdamW(
            self.discriminator.parameters(), **self.config.optim.optimizer
        )

        generator_scheduler = get_scheduler(
            optimizer=generator_optimizer, **self.config.optim.scheduler
        )
        discriminator_scheduler = get_scheduler(
            optimizer=discriminator_optimizer, **self.config.optim.scheduler
        )

        generator_optimizer = {"optimizer": generator_optimizer, "frequency": 1}
        discriminator_optimizer = {
            "optimizer": discriminator_optimizer,
            "frequency": self.config.optim.num_discriminator_steps,
        }

        generator_scheduler = {"scheduler": generator_scheduler, "interval": "step"}
        discriminator_scheduler = {
            "scheduler": discriminator_scheduler,
            "interval": "step",
        }

        return (
            [generator_optimizer, discriminator_optimizer],
            [generator_scheduler, discriminator_scheduler],
        )

    def on_load_checkpoint(self, checkpoint: dict[str, Any]):
        if "ApexMixedPrecisionPlugin" in checkpoint:
            checkpoint.pop("ApexMixedPrecisionPlugin")
