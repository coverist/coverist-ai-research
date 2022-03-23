import os
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class BigGANDataset(Dataset):
    def __init__(
        self,
        dataset: pd.DataFrame,
        imagedir: str,
        label2id: dict[str, int],
        transform: Callable[[np.ndarray], torch.Tensor],
    ):
        self.dataset = dataset
        self.imagedir = imagedir
        self.label2id = label2id
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.dataset.iloc[index]
        filename = os.path.join(self.imagedir, *row.barcode[-3:], f"{row.barcode}.jpg")

        image = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
        image = self.transform(image=image)["image"]
        label = torch.tensor(self.label2id[row.category], dtype=torch.long)

        return image, label
