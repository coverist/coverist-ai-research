import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Optional

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
    negative_samples: int
    transform: Callable
    tokenizer: PreTrainedTokenizerBase

    def __post_init__(self):
        self.turbojpeg = TurboJPEG()

    def __len__(self) -> int:
        return len(self.dataset)

    def read_image_and_transform(self, example: pd.Series) -> Optional[torch.Tensor]:
        path = os.path.join(self.image_dir, *example.isbn[-3:], f"{example.isbn}.jpg")
        try:
            with open(path, "rb") as fp:
                image = self.turbojpeg.decode(fp.read(), pixel_format=TJCS_RGB)
                return self.transform(image=image)["image"]
        except OSError:
            # Sometimes when the image is truncated, turbojpeg will throw `OSError`.
            # This function will return `None` to notify that the current example has
            # invalid image.
            return None

    def create_text_encoding(
        self, example: pd.Series, negative: bool = False
    ) -> dict[str, Any]:
        queries = [example.title, example.author, example.publisher]
        negative_queries = queries.copy()
        target_index = random.randint(0, len(queries) - 1)

        # Create new negative sample which has different query from the original one.
        # Note that if `negative=False` then nothing will be replaced.
        while negative and negative_queries[target_index] == queries[target_index]:
            alternatvie = self.dataset.sample(1).iloc[0]
            alternatvie = [alternatvie.title, alternatvie.author, alternatvie.publisher]
            negative_queries[target_index] = alternatvie[target_index]

        return self.tokenizer(
            f" {self.tokenizer.sep_token} ".join(negative_queries),
            truncation=True,
            max_length=self.max_length,
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, list[dict[str, Any]]]:
        example = self.dataset.iloc[index]
        image = self.read_image_and_transform(example)
        if image is None:
            return self[random.randint(0, len(self) - 1)]

        # Create one positive text sample and several negative samples. All samples will
        # be merged and compared with the images.
        text_encodings = [
            self.create_text_encoding(example, negative=i > 0)
            for i in range(1 + self.negative_samples)
        ]
        return image, text_encodings


@dataclass
class DataCollatorForImageTextPair(DataCollatorWithPadding):
    def __call__(
        self, features: list[tuple[torch.Tensor, dict[str, Any]]]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        images = [feature[0] for feature in features]
        encodings = [encoding for feature in features for encoding in feature[1]]

        return torch.stack(images), dict(super().__call__(encodings))
