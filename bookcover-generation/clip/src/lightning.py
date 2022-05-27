import os
from typing import Any, Optional

import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule
from sklearn.model_selection import train_test_split
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)

from dataset import BookCoverPairedDataset, CLIPTransform, DataCollatorForImageTextPair

try:
    from apex.optimizers import FusedAdam as AdamW
except ModuleNotFoundError:
    from torch.optim import AdamW


class CLIPTrainingModule(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        self.image_encoder = timm.create_model(
            config.model.image_encoder,
            pretrained=True,
            num_classes=config.model.embedding_dim,
        )
        self.text_encoder = AutoModelForSequenceClassification.from_pretrained(
            config.model.text_encoder,
            num_labels=config.model.embedding_dim,
        )
        self.temperature = nn.Parameter(torch.tensor(2.6593))

        # If `use_gradient_checkpoint=True` then enable the gradient checkpointing to
        # the image-encoder and text-encoder model to save the GPU memory usage.
        if config.train.use_gradient_checkpoint:
            self.image_encoder.set_grad_checkpointing()
            self.text_encoder.gradient_checkpointing_enable()

    def forward(
        self,
        input_images: torch.Tensor,
        input_texts: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        # Get image embeddings and text embeddings to calculate cosine similarity.
        image_features = F.normalize(self.image_encoder(input_images), eps=1e-6)
        text_features = F.normalize(self.text_encoder(**input_texts)[0], eps=1e-6)

        # Calculate the cosine similarity and scale the logits.
        logits = torch.matmul(image_features, text_features.transpose(0, 1))
        logits = logits * self.temperature.exp().clamp_max(100)
        labels = torch.arange(logits.size(0), device=logits.device)

        # Calculate image-to-text and text-to-image cross-entropy loss and accuracy.
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.transpose(0, 1), labels)
        accuracy_i2t = (logits.argmax(dim=1) == labels).float().mean()
        accuracy_t2i = (logits.argmax(dim=0) == labels).float().mean()

        return {
            "loss": (loss_i2t + loss_t2i) / 2,
            "loss_i2t": loss_i2t,
            "loss_t2i": loss_t2i,
            "accuracy_i2t": accuracy_i2t,
            "accuracy_t2i": accuracy_t2i,
        }

    def training_step(self, batch: tuple[torch.Tensor, ...], idx: int) -> torch.Tensor:
        metrics = self(*batch)
        self.log("step", self.global_step)
        self.log_dict({f"train/{k}": v for k, v in metrics.items()})
        return metrics["loss"]

    def validation_step(self, batch: tuple[torch.Tensor, ...], idx: int):
        metrics = self(*batch)
        self.log("step", self.global_step)
        self.log_dict({f"val/{k}": v for k, v in metrics.items()})

    def get_parameter_groups(self) -> list[dict[str, Any]]:
        do_decay = [p for p in self.parameters() if p.ndim < 2]
        no_decay = [p for p in self.parameters() if p.ndim >= 2]
        return [{"params": do_decay}, {"params": no_decay, "weight_decay": 0.0}]

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        optimizer = AdamW(self.get_parameter_groups(), **self.config.optim.optimizer)
        scheduler = get_scheduler(optimizer=optimizer, **self.config.optim.scheduler)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def on_load_checkpoint(self, checkpoint: dict[str, Any]):
        if "ApexMixedPrecisionPlugin" in checkpoint:
            checkpoint.pop("ApexMixedPrecisionPlugin")


class CLIPDataModule(LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

    def setup(self, stage: Optional[str] = None):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.text_encoder)

        # Load the book dataset and filter invalid samples. We will only use the images
        # of which aspect ratios are more than `0.5` and less than `0.9`.
        dataset = pd.read_json(self.config.data.dataset, lines=True, dtype=False)
        dataset = dataset[dataset.with_cover]
        dataset = dataset[dataset.cover_aspect_ratio > 0.5]
        dataset = dataset[dataset.cover_aspect_ratio < 0.9]

        train_dataset, val_dataset = train_test_split(
            dataset,
            test_size=self.config.data.validation_ratio,
            random_state=42,
            shuffle=True,
        )
        self.train_dataset = BookCoverPairedDataset(
            train_dataset,
            image_dir=self.config.data.image_dir,
            max_length=self.config.data.max_length,
            drop_prob=self.config.data.drop_prob,
            transform=CLIPTransform(self.config.data.image_size, augmentation=True),
            tokenizer=tokenizer,
        )
        self.val_dataset = BookCoverPairedDataset(
            val_dataset,
            image_dir=self.config.data.image_dir,
            max_length=self.config.data.max_length,
            drop_prob=0.0,
            transform=CLIPTransform(self.config.data.image_size, augmentation=False),
            tokenizer=tokenizer,
        )
        self.collator = DataCollatorForImageTextPair(tokenizer, pad_to_multiple_of=8)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            collate_fn=self.collator,
            prefetch_factor=1,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.train.batch_size,
            num_workers=os.cpu_count(),
            collate_fn=self.collator,
            prefetch_factor=1,
            persistent_workers=True,
        )
