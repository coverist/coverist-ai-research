import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Optional

import albumentations as A
import cv2
import numpy as np
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


class BookCoverPairedDataset(Dataset):
    def __init__(
        self,
        dataset: pd.DataFrame,
        image_dir: str,
        max_length: int,
        drop_prob: float,
        transform: Callable,
        tokenizer: PreTrainedTokenizerBase,
    ):
        # It is observed that `pd.DataFrame` is really slower than `np.array`. Thus we
        # convert the dataframe to numpy array to reduce the bottleneck.
        self.dataset = dataset.to_numpy()
        self.image_dir = image_dir
        self.max_length = max_length
        self.drop_prob = drop_prob
        self.transform = transform
        self.tokenizer = tokenizer

        # Instead of using `cv2.imread`, we use `TurboJPEG` of `libturbo-jpeg` to
        # improve image decoding performance.
        self.turbojpeg = TurboJPEG()

    def __len__(self) -> int:
        return len(self.dataset)

    def read_image_and_transform(self, example: np.ndarray) -> Optional[torch.Tensor]:
        path = os.path.join(self.image_dir, *example[6][-3:], f"{example[6]}.jpg")
        try:
            with open(path, "rb") as fp:
                image = self.turbojpeg.decode(fp.read(), pixel_format=TJCS_RGB)
                return self.transform(image=image)["image"]
        except OSError:
            # Sometimes when the image is truncated, turbojpeg will throw `OSError`.
            # This function will return `None` to notify that the current example has
            # invalid image.
            return None

    def create_text_encoding(self, example: np.ndarray) -> dict[str, Any]:
        text_queries = [
            f"제목: {example[0]}",
            f"저자: {example[2]}",
            f"출판사: {example[4]}",
        ]
        if self.drop_prob > 0:
            random.shuffle(text_queries)

        # Drop the queries randomly to make the model to see different combinations.
        for i in reversed(range(len(text_queries))):
            if len(text_queries) > 1 and random.random() < self.drop_prob:
                text_queries = text_queries[:i] + text_queries[i + 1 :]

        return self.tokenizer(
            " ___ ".join(text_queries),
            truncation=True,
            max_length=self.max_length,
        )

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, Any]]:
        example = self.dataset[index]
        image = self.read_image_and_transform(example)
        encoding = self.create_text_encoding(example)

        if image is None:
            # If the image is invalid, randomly selected other sample will be returned.
            return self[random.randint(0, len(self) - 1)]
        return image, encoding


@dataclass
class DataCollatorForImageTextPair(DataCollatorWithPadding):
    def __call__(
        self, features: list[tuple[torch.Tensor, dict[str, Any]]]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        images = [feature[0] for feature in features]
        encodings = [feature[1] for feature in features]

        return torch.stack(images), dict(super().__call__(encodings))
