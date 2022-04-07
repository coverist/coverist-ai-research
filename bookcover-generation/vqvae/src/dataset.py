import glob

import cv2
import torch
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, pattern: str, resolution: int = 256):
        self.filenames = glob.glob(pattern, recursive=True)
        self.resolution = resolution

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, index: str) -> torch.Tensor:
        image = cv2.imread(self.filenames[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.resolution, self.resolution))

        image = torch.from_numpy(image)
        image = 2 * image.permute(2, 0, 1).float() / 0xFF - 1
        return image
