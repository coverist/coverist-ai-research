import os
from typing import Any, Optional

import pandas as pd
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule, LightningModule
from sklearn.model_selection import train_test_split
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EncoderDecoderModel,
    GPT2Config,
    GPT2LMHeadModel,
    get_scheduler,
)

from dataset import DALLEBookDataset
from modeling import VQVAEDecoder

try:
    from apex.optimizers import FusedAdam as AdamW
except ModuleNotFoundError:
    from torch.optim import AdamW


class DALLETrainingModule(LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            config.model.encoder,
            decoder_model=GPT2LMHeadModel(GPT2Config(**config.model.decoder)),
        )
        self.model.config.decoder_start_token_id = config.model.decoder.bos_token_id
        self.model.config.eos_token_id = config.model.decoder.eos_token_id
        self.model.config.pad_token_id = config.model.decoder.bos_token_id
        self.model.config.vocab_size = config.model.decoder.vocab_size

        self.vqgan = VQVAEDecoder.from_pretrained(config.model.vqgan)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.encoder)

    def training_step(self, batch: dict[str, torch.Tensor], idx: int) -> torch.Tensor:
        loss = self.model(**batch).loss
        self.log("train/loss", loss)
        self.log("step", self.global_step)
        return loss

    def validation_step(
        self, batch: dict[str, torch.Tensor], idx: int
    ) -> dict[str, torch.Tensor]:
        self.log("val/loss", self.model(**batch).loss)
        return batch

    def validation_epoch_end(self, batch_list: list[dict[str, torch.Tensor]]):
        outputs = self.model.generate(
            batch_list[0]["input_ids"],
            attention_mask=batch_list[0]["attention_mask"],
            **self.config.model.generation
        )
        outputs = outputs[:, 1:].view(outputs.size(0), int(outputs.size(1) ** 0.5), -1)

        images = []
        for i in range(0, outputs.size(0), 64):
            images.extend(self.vqgan(outputs[i : i + 64]))

        self.logger.log_image(
            "val/generated",
            images=images,
            caption=self.tokenizer.batch_decode(batch_list[0]["input_ids"], True),
        )

    def get_parameter_groups(self) -> list[dict[str, Any]]:
        do_decay = [p for p in self.model.parameters() if p.ndim < 2]
        no_decay = [p for p in self.model.parameters() if p.ndim >= 2]
        return [{"params": do_decay}, {"params": no_decay, "weight_decay": 0.0}]

    def configure_optimizers(self) -> tuple[list[Optimizer], list[dict[str, Any]]]:
        optimizer = AdamW(self.get_parameter_groups(), **self.config.optim.optimizer)
        scheduler = get_scheduler(optimizer=optimizer, **self.config.optim.scheduler)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def on_load_checkpoint(self, checkpoint: dict[str, Any]):
        if "ApexMixedPrecisionPlugin" in checkpoint:
            checkpoint.pop("ApexMixedPrecisionPlugin")


class DALLETrainingDataModule(LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

    def setup(self, stage: Optional[str] = None):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model.encoder)

        books = pd.read_json(self.config.dataset.text_dataset, lines=True, dtype=None)
        books = books[books.with_cover]
        books = books[books.cover_aspect_ratio > 0.5]
        books = books[books.cover_aspect_ratio < 0.9]

        images = pd.read_csv(
            self.config.dataset.image_dataset, dtype={"id": str}, index_col="id"
        )
        dataset = DALLEBookDataset(
            tokenizer=tokenizer,
            books=books,
            images=images,
            max_length=self.config.dataset.prompt_max_length,
            image_vocab_size=self.config.dataset.image_vocab_size,
        )

        train_indices, val_indices = train_test_split(
            range(len(dataset)),
            test_size=self.config.dataset.validation_ratio,
            random_state=42,
            shuffle=True,
        )
        self.train_dataset = Subset(dataset, train_indices)
        self.val_dataset = Subset(dataset, val_indices)
        self.collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            self.config.train.batch_size,
            shuffle=True,
            num_workers=os.cpu_count(),
            collate_fn=self.collator,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            self.config.train.batch_size,
            num_workers=os.cpu_count(),
            collate_fn=self.collator,
            persistent_workers=True,
        )
