import os
import random
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from omegaconf import DictConfig
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset


@dataclass
class VQGANDecoderDataset(Dataset):
    image_directory: str
    images_index: list[str]
    images_latents: np.ndarray
    image_size: int

    def __len__(self) -> int:
        return len(self.images_index)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        filename = self.images_index[index]
        filename = os.path.join(self.image_directory, *filename[-3:], filename + ".jpg")
        image = cv2.imread(filename)

        # There are some images which are truncated. Because `cv2` cannot read the
        # invalid images and return `None`, we will replace current sample to other
        # image from the dataset.
        if image is None:
            return self[random.randint(0, len(self) - 1)]

        # Convert the color from BGR to RGB, resize the image, and normalize the image
        # with wrapping by `torch.tensor`.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(
            image,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_AREA,
        )
        image = 2 * torch.from_numpy(image).float().permute(2, 0, 1) / 0xFF - 1

        return torch.from_numpy(self.images_latents[index]), image


def create_train_val_dataloaders(config: DictConfig) -> tuple[DataLoader, DataLoader]:
    with open(config.data.quantized_index, "r") as fp:
        images_index = fp.read().splitlines()
    images_latents = np.load(config.data.quantized_images, mmap_mode="r")

    dataset = VQGANDecoderDataset(
        config.data.image_directory,
        images_index,
        images_latents,
        config.data.image_size,
    )
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
        pin_memory=False,
        persistent_workers=True,
    )
    val_dataloader = DataLoader(
        dataset=Subset(dataset, val_indices),
        batch_size=config.train.batch_size,
        num_workers=os.cpu_count(),
        pin_memory=False,
        persistent_workers=True,
    )
    return train_dataloader, val_dataloader
