import os
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np
import pandas as pd
import torch
from scipy.stats import truncnorm
from torch.utils.data import Dataset


@dataclass
class BigGANImageDataset(Dataset):
    dataset: pd.DataFrame
    image_dir: str
    image_size: int
    labels: list[str]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.dataset.iloc[index]
        filename = os.path.join(self.image_dir, *item.isbn[-3:], f"{item.isbn}.jpg")

        image = cv2.imread(filename)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))

        return {
            "images": torch.from_numpy(2 * image.astype(np.float32) / 0xFF - 1),
            "labels": torch.tensor(self.labels.index(item.category), dtype=torch.long),
        }


@dataclass
class BigGANRandomDataset(Dataset):
    dataset: pd.DataFrame
    latent_dim: int
    num_labels: int
    num_samples: int
    truncation: Optional[float] = None

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        latent = (
            truncnorm.rvs(-self.truncation, self.truncation, size=self.latent_dim)
            if self.truncation is not None
            else np.random.randn(self.latent_dim)
        )
        return {
            "latents": torch.from_numpy(latent),
            "labels": torch.randint(0, self.num_labels, size=(), dtype=torch.long),
        }
