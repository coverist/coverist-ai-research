import glob
import os
import random

import cv2
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset


class RecursiveImageDataset(Dataset):
    def __init__(self, image_files: str, image_size: int = 256):
        self.filenames = glob.glob(image_files, recursive=True)
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: str) -> torch.Tensor:
        image = cv2.imread(self.filenames[index])

        # There are some images which are truncated. Because `cv2` cannot read the
        # invalid images and return `None`, we will replace current sample to other
        # image from the dataset.
        if image is None:
            return self[random.randint(0, len(self) - 1)]

        # Convert the color from BGR to RGB and resize the image.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )

        # Normalize the image and wrap with `torch.tensor`.
        image = torch.from_numpy(2 * image.transpose(2, 0, 1) / 0xFF - 1)
        return image


def create_train_val_dataloaders(config: DictConfig) -> tuple[DataLoader, DataLoader]:
    dataset = RecursiveImageDataset(config.data.image_files, config.data.image_size)
    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        test_size=config.data.validation_ratio,
        random_state=42,
        shuffle=True,
    )

    train_dataloader = DataLoader(
        dataset=Subset(dataset, train_indices),
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=os.cpu_count(),
        pin_memory=True,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        dataset=Subset(dataset, val_indices),
        batch_size=config.train.batch_size,
        num_workers=os.cpu_count(),
        pin_memory=True,
        persistent_workers=True,
    )
    return train_dataloader, val_dataloader
