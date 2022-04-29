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
        self.negative_samples = config.data.negative_samples

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get image embeddings and text embeddings to calculate cosine similarity.
        image_features = F.normalize(self.image_encoder(input_images), eps=1e-6)
        text_features = F.normalize(self.text_encoder(**input_texts)[0], eps=1e-6)

        # Calculate the cosine similarity and scale the logits.
        logits = torch.matmul(image_features, text_features.transpose(0, 1))
        logits = logits * self.temperature.exp().clamp_max(100)

        # Calculate cross-entropy loss and accuracy for entire correctness of
        # predictions. Note that each image example has one positive text sample and
        # several negative samples.
        labels = torch.arange(
            start=0,
            end=logits.size(1),
            step=1 + self.negative_samples,
            device=logits.device,
        )
        loss = F.cross_entropy(logits, labels)
        accuracy = (logits.argmax(dim=1) == labels).float().mean()

        # Compute the inter-group accuracy and groupwise accuracy.
        grouped_logits = logits.unflatten(-1, (-1, 1 + self.negative_samples))
        groupwise_accuracy = (grouped_logits.argmax(dim=2) == 0).float().mean()

        labels = torch.arange(0, logits.size(0), device=logits.device)
        intergroup_accuracy = grouped_logits.max(dim=2).values.argmax(dim=1) == labels
        intergroup_accuracy = intergroup_accuracy.float().mean()

        return loss, accuracy, intergroup_accuracy, groupwise_accuracy

    def training_step(self, batch: tuple[torch.Tensor, ...], idx: int) -> torch.Tensor:
        loss, accuracy, intergroup_accuracy, groupwise_accuracy = self(*batch)
        self.log("train/loss", loss)
        self.log("train/accuracy", accuracy)
        self.log("train/intergroup_accuracy", intergroup_accuracy)
        self.log("train/groupwise_accuracy", groupwise_accuracy)
        self.log("step", self.global_step)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, ...], idx: int):
        loss, accuracy, intergroup_accuracy, groupwise_accuracy = self(*batch)
        self.log("val/loss", loss)
        self.log("val/accuracy", accuracy)
        self.log("val/intergroup_accuracy", intergroup_accuracy)
        self.log("val/groupwise_accuracy", groupwise_accuracy)
        self.log("step", self.global_step)

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
            negative_samples=self.config.data.negative_samples,
            transform=CLIPTransform(self.config.data.image_size, augmentation=True),
            tokenizer=tokenizer,
        )
        self.val_dataset = BookCoverPairedDataset(
            val_dataset,
            image_dir=self.config.data.image_dir,
            max_length=self.config.data.max_length,
            negative_samples=self.config.data.negative_samples,
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
