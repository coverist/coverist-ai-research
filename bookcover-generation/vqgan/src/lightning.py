from typing import Any

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torchvision.utils import make_grid

from modeling import (
    OCRPerceptualExtractor,
    PatchDiscriminator,
    PatchDiscriminatorConfig,
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


class VQGANTrainingModule(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.loss_perceptual_weights = config.optim.loss_perceptual_weights
        self.loss_generator_weight = config.optim.loss_generator_weight
        self.use_gan_after = config.optim.use_gan_after

        self.encoder = VQVAEEncoder(VQVAEEncoderConfig(**config.model.encoder))
        self.decoder = VQVAEDecoder(VQVAEDecoderConfig(**config.model.decoder))
        self.quantizer = VQVAEQuantizer(VQVAEQuantizerConfig(**config.model.quantizer))

        self.discriminator = PatchDiscriminator(
            PatchDiscriminatorConfig(**config.model.discriminator)
        )
        self.perceptual = OCRPerceptualExtractor()

    def generator_step(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        encoded = self.encoder(images)
        latents = self.quantizer(encoded)
        decoded = self.decoder(encoded + (latents - encoded).detach())

        loss_perceptual_list = list(
            map(F.l1_loss, self.perceptual(images), self.perceptual(decoded))
        )
        loss_perceptual = sum(
            weight * loss
            for weight, loss in zip(self.loss_perceptual_weights, loss_perceptual_list)
        )
        loss_reconstruction = F.l1_loss(images, decoded)
        loss_quantization = F.l1_loss(encoded, latents)

        if self.current_epoch < self.use_gan_after:
            loss_generator = -self.discriminator(decoded).mean()
        else:
            loss_generator = 0

        loss = (
            loss_reconstruction
            + loss_perceptual
            + loss_quantization
            + self.loss_generator_weight * loss_generator
        )
        metrics = {
            "loss_reconstruction": loss_reconstruction,
            "loss_perceptual": loss_perceptual,
            "loss_quantization": loss_quantization,
            "loss_generator": loss_generator,
        }
        return decoded, loss, metrics

    def discriminator_step(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self.current_epoch < self.use_gan_after:
            return None, None, {}

        decoded = self.decoder(self.quantizer(self.encoder(images)))
        loss_discriminator_real = (1 - self.discriminator(images)).relu().mean()
        loss_discriminator_fake = (1 + self.discriminator(decoded)).relu().mean()

        loss_discriminator = loss_discriminator_real + loss_discriminator_fake
        return decoded, loss_discriminator, {"loss_discriminator": loss_discriminator}

    def forward(
        self, images: torch.Tensor, optimizer_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        if self.training:
            self.encoder.requires_grad_(optimizer_idx == 0)
            self.decoder.requires_grad_(optimizer_idx == 0)
            self.quantizer.requires_grad_(optimizer_idx == 0)
            self.discriminator.requires_grad_(optimizer_idx == 1)

            self.encoder.train(optimizer_idx == 0)
            self.decoder.train(optimizer_idx == 0)
            self.quantizer.train(optimizer_idx == 0)
            self.discriminator.train(optimizer_idx == 1)

        if optimizer_idx == 0:
            return self.generator_step(images)
        elif optimizer_idx == 1:
            return self.discriminator_step(images)

    def training_step(
        self, images: torch.Tensor, batch_idx: int, optimizer_idx: int
    ) -> torch.Tensor:
        _, loss, metrics = self(images, int(optimizer_idx))
        self.log("step", self.global_step)
        self.log_dict({f"train/{k}": v for k, v in metrics.items()})
        return loss

    def validation_step(
        self, images: torch.Tensor, batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        decoded, _, metrics = self(images, optimizer_idx=0)
        self.log("step", self.global_step)
        self.log_dict({f"val/{k}": v for k, v in metrics.items()})
        return images, decoded

    def validation_epoch_end(self, outputs: list[tuple[torch.Tensor, torch.Tensor]]):
        images = torch.stack(outputs[0], dim=1).flatten(0, 1)
        nrow = int(images.size(0) ** 0.5) // 2 * 2
        grid = make_grid(images, nrow, value_range=(-1, 1))
        self.logger.log_image("val/reconstructed", [grid])

    def configure_optimizers(self) -> tuple[dict[str, Any]]:
        generator_params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.quantizer.parameters())
        )
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
