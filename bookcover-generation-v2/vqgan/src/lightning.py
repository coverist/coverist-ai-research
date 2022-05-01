import math
from typing import Any, Optional

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning import LightningModule
from torchvision.utils import make_grid

from modeling import (
    PatchDiscriminator,
    PerceptualLoss,
    VQGANDecoder,
    VQGANEncoder,
    VQGANQuantizer,
)

try:
    from apex.optimizers import FusedAdam as Adam
except ModuleNotFoundError:
    from torch.optim import Adam


class VQGANTrainingModule(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.num_log_batches = math.ceil(64 / self.config.train.batch_size)

        self.perceptual_loss_weight = config.optim.criterion.perceptual
        self.quantization_loss_weight = config.optim.criterion.quantization
        self.adversarial_loss_weight = config.optim.criterion.adversarial

        self.encoder = VQGANEncoder(**config.model.encoder)
        self.decoder = VQGANDecoder(**config.model.decoder)
        self.quantizer = VQGANQuantizer(**config.model.quantizer)
        self.discriminator = PatchDiscriminator(**config.model.discriminator)
        self.perceptual = PerceptualLoss(**config.model.perceptual)

    def calculate_adaptive_weight(
        self, content: torch.Tensor, adversarial: torch.Tensor
    ) -> torch.Tensor:
        # If this function is called when validating, then just return `0` because it is
        # impossible to calculate the gradients in validating mode.
        if not self.training:
            return 0

        last_layer = self.decoder.head.parametrizations.weight.original
        grad_cnt = torch.autograd.grad(content, last_layer, retain_graph=True)[0]
        grad_adv = torch.autograd.grad(adversarial, last_layer, retain_graph=True)[0]
        return (grad_cnt.norm() / (grad_adv.norm() + 1e-4)).clamp(0, 1e4).detach()

    def generator_step(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        # Encode the images, quantize the latents, and decode with quantized vectors.
        encodings = self.encoder(images)
        latents, _, loss_quantization, perplexity = self.quantizer(encodings)
        decodings = self.decoder(latents)

        # Calculate the reconstruction loss, perceptual loss, and adversarial loss.
        loss_reconstruction = F.l1_loss(images, decodings)
        loss_perceptual = self.perceptual(images, decodings)
        loss_generator = -self.discriminator(decodings).mean()

        # Note that we will use adaptive adversarial loss which is calculated by
        # estimating the ratio of gradient norms of the last-layer weight.
        adaptive_weight = self.calculate_adaptive_weight(
            loss_reconstruction + self.perceptual_loss_weight * loss_perceptual,
            loss_generator,
        )
        loss = (
            loss_reconstruction
            + self.perceptual_loss_weight * loss_perceptual
            + self.quantization_loss_weight * loss_quantization
            + self.adversarial_loss_weight * adaptive_weight * loss_generator
        )
        metrics = {
            "loss_reconstruction": loss_reconstruction,
            "loss_perceptual": loss_perceptual,
            "loss_quantization": loss_quantization,
            "loss_generator": loss_generator,
            "adaptive_weight": adaptive_weight,
            "perplexity": perplexity,
            "encoding_norm": encodings.norm(dim=1).mean(),
        }
        return decodings, loss, metrics

    def discriminator_step(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        decodings = self.decoder(self.quantizer(self.encoder(images))[0])
        loss_discriminator_real = (1 - self.discriminator(images)).relu().mean()
        loss_discriminator_fake = (1 + self.discriminator(decodings)).relu().mean()
        loss_discriminator = loss_discriminator_real + loss_discriminator_fake

        metrics = {
            "loss_discriminator": loss_discriminator,
            "loss_discriminator_real": loss_discriminator_real,
            "loss_discriminator_fake": loss_discriminator_fake,
        }
        return decodings, loss_discriminator, metrics

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
    ) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        decodings, _, metrics = self(images, optimizer_idx=0)
        self.log("step", self.global_step)
        self.log_dict({f"val/{k}": v for k, v in metrics.items()})

        if batch_idx < self.num_log_batches:
            return images, decodings
        return None, None

    def validation_epoch_end(
        self, outputs: list[tuple[Optional[torch.Tensor], Optional[torch.Tensor]]]
    ):
        # Get 64 original and reconstructed images.
        outputs = [output for output in outputs if output[0] is not None]
        images = torch.cat([output[0] for output in outputs])[:64]
        decodings = torch.cat([output[1] for output in outputs])[:64]

        # Stack the original and reconstructed images, make a grid, and log the image.
        grid = torch.stack((images, decodings), dim=1).flatten(0, 1)
        grid = make_grid(grid, value_range=(-1, 1))
        self.logger.log_image("val/reconstructed", [grid])

    def configure_optimizers(self) -> tuple[dict[str, Any], dict[str, Any]]:
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
