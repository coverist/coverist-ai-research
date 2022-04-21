from typing import Any, Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torch.optim import Optimizer
from torchvision.utils import make_grid

from modeling import (
    OCRPerceptualLoss,
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

        self.decoder_ema_decay = config.optim.decoder_ema_decay
        self.model_average_start = config.optim.model_average_start
        self.adversarial_start = config.optim.adversarial_start

        self.perceptual_weight = config.optim.criterion.perceptual_weight
        self.quantization_weight = config.optim.criterion.quantization_weight
        self.generator_weight = config.optim.criterion.generator_weight

        self.encoder = VQVAEEncoder(VQVAEEncoderConfig(**config.model.encoder))
        self.decoder = VQVAEDecoder(VQVAEDecoderConfig(**config.model.decoder))
        self.decoder_ema = VQVAEDecoder(VQVAEDecoderConfig(**config.model.decoder))
        self.quantizer = VQVAEQuantizer(VQVAEQuantizerConfig(**config.model.quantizer))

        self.discriminator = PatchDiscriminator(
            PatchDiscriminatorConfig(**config.model.discriminator)
        )
        self.perceptual = OCRPerceptualLoss(
            tuple(config.optim.criterion.perceptual_input_size)
        )

    def generator_step(
        self, images: torch.Tensor, use_ema: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        encoded = self.encoder(images)
        latents, _, loss_quantization, perplexity = self.quantizer(encoded)
        decoded = self.decoder_ema(latents) if use_ema else self.decoder(latents)

        loss_reconstruction = F.l1_loss(images, decoded)
        loss_perceptual = self.perceptual(images, decoded)

        loss_generator = 0
        if self.current_epoch >= self.adversarial_start:
            loss_generator = -self.discriminator(decoded).mean()

        loss = (
            loss_reconstruction
            + self.perceptual_weight * loss_perceptual
            + self.quantization_weight * loss_quantization
            + self.generator_weight * loss_generator
        )
        metrics = {
            "loss_reconstruction": loss_reconstruction,
            "loss_perceptual": loss_perceptual,
            "loss_quantization": loss_quantization,
            "loss_generator": loss_generator,
            "encoding_norms": encoded.norm(dim=1).mean(),
            "perplexity": perplexity,
        }
        return decoded, loss, metrics

    def discriminator_step(
        self, images: torch.Tensor, use_ema: bool = False
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], dict[str, torch.Tensor]]:
        if self.current_epoch < self.adversarial_start:
            return None, None, {"loss_discriminator": 0}

        decoded = self.decoder(self.quantizer(self.encoder(images))[0])
        loss_discriminator_real = (1 - self.discriminator(images)).relu().mean()
        loss_discriminator_fake = (1 + self.discriminator(decoded)).relu().mean()

        loss_discriminator = loss_discriminator_real + loss_discriminator_fake
        return decoded, loss_discriminator, {"loss_discriminator": loss_discriminator}

    def forward(
        self, images: torch.Tensor, optimizer_idx: int, use_ema: bool = False
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor], dict[str, torch.Tensor]]:
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
            return self.generator_step(images, use_ema)
        elif optimizer_idx == 1:
            return self.discriminator_step(images, use_ema)

    def training_step(
        self, images: torch.Tensor, batch_idx: int, optimizer_idx: int
    ) -> Optional[torch.Tensor]:
        _, loss, metrics = self(images, int(optimizer_idx))
        self.log("step", self.global_step)
        self.log_dict({f"train/{k}": v for k, v in metrics.items()})
        return loss

    def validation_step(
        self, images: torch.Tensor, batch_idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        decoded, _, metrics = self(images, optimizer_idx=0)
        decoded_ema, _, metrics_ema = self(images, optimizer_idx=0, use_ema=True)
        self.log("step", self.global_step)
        self.log_dict({f"val/{k}": v for k, v in metrics.items()})
        self.log_dict({f"val_ema/{k}": v for k, v in metrics_ema.items()})
        return images, decoded, decoded_ema

    def validation_epoch_end(
        self, outputs: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    ):
        images = torch.stack((outputs[0][0], outputs[0][1]), dim=1).flatten(0, 1)
        images_ema = torch.stack((outputs[0][0], outputs[0][2]), dim=1).flatten(0, 1)

        nrow = int(images.size(0) ** 0.5) // 2 * 2
        grid = make_grid(images.clamp(-1, 1), nrow, value_range=(-1, 1))
        grid_ema = make_grid(images_ema.clamp(-1, 1), nrow, value_range=(-1, 1))

        self.logger.log_image("val/reconstructed", [grid])
        self.logger.log_image("val_ema/reconstructed", [grid_ema])

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

    @torch.no_grad()
    def on_before_optimizer_step(self, optimizer: Optimizer, optimizer_idx: int):
        if optimizer_idx != 0:
            return

        decay = 0.0
        if self.current_epoch >= self.model_average_start:
            decay = self.decoder_ema_decay

        for p1, p2 in zip(self.decoder.parameters(), self.decoder_ema.parameters()):
            p2.copy_(decay * p2.float() + (1 - decay) * p1.float())
        for b1, b2 in zip(self.decoder.buffers(), self.decoder_ema.buffers()):
            b2.copy_(b1)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]):
        if "ApexMixedPrecisionPlugin" in checkpoint:
            checkpoint.pop("ApexMixedPrecisionPlugin")
