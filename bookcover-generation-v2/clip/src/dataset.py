import os
import random
from dataclasses import dataclass
from typing import Any, Callable

import albumentations as A
import cv2
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding, PreTrainedTokenizerBase
from turbojpeg import TJCS_RGB, TurboJPEG


class CLIPTransform(A.Compose):
    def __init__(self, image_size: int, augmentation: bool = True):
        transforms = [
            A.Resize(image_size, image_size),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5 if augmentation else 0.0,
            ),
            A.ShiftScaleRotate(
                shift_limit=0.1,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_CONSTANT,
                value=0x7F,
                p=0.5 if augmentation else 0.0,
            ),
            A.Cutout(
                num_holes=1,
                max_h_size=96,
                max_w_size=96,
                fill_value=0x7F,
                p=0.2 if augmentation else 0.0,
            ),
            A.Normalize(0.5, 0.5),
            ToTensorV2(),
        ]
        super().__init__(transforms)


@dataclass
class BookCoverPairedDataset(Dataset):
    dataset: pd.DataFrame
    image_dir: str
    max_length: int
    transform: Callable
    tokenizer: PreTrainedTokenizerBase

    def __post_init__(self):
        self.turbojpeg = TurboJPEG()

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        example = self.dataset.iloc[index]
        path = os.path.join(self.image_dir, *example.isbn[-3:], f"{example.isbn}.jpg")

        # Read the book-cover image and apply augmentations.
        try:
            with open(path, "rb") as fp:
                image = self.turbojpeg.decode(fp.read(), pixel_format=TJCS_RGB)
                image = self.transform(image=image)["image"]
        except OSError:
            return self[random.randint(0, len(self))]

        # Create the text query prompt and tokenize with truncation.
        text_queries = [example.title, example.author, example.publisher]
        text_encoding = self.tokenizer(
            f" {self.tokenizer.sep_token} ".join(text_queries),
            truncation=True,
            max_length=self.max_length,
        )
        return image, text_encoding


@dataclass
class DataCollatorForImageTextPair(DataCollatorWithPadding):
    def __call__(
        self, features: list[tuple[torch.Tensor, dict[str, Any]]]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        images = [feature[0] for feature in features]
        encodings = [feature[1] for feature in features]

        return torch.stack(images), super().__call__(encodings)
