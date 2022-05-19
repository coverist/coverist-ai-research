import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


@dataclass
class DALLEBookDataset(Dataset):
    tokenizer: PreTrainedTokenizerBase
    books: pd.DataFrame
    images_index: list[str]
    images_list: list[np.ndarray]
    max_length: int

    def __len__(self) -> int:
        return len(self.books)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        book_example = self.books.iloc[index]
        image_example = random.choice(self.images_list)
        image_example = image_example[self.images_index.index(book_example.isbn)]

        description = [
            book_example.title,
            book_example.author,
            book_example.publisher,
            book_example.category,
        ]
        description = f" {self.tokenizer.sep_token} ".join(description)

        batch = self.tokenizer(description, truncation=True, max_length=self.max_length)
        batch["labels"] = torch.from_numpy(image_example)
        return batch
