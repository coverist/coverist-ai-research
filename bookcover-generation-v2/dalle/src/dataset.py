from dataclasses import dataclass

import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


@dataclass
class DALLEBookDataset(Dataset):
    tokenizer: PreTrainedTokenizerBase
    books: pd.DataFrame
    images: pd.DataFrame
    max_length: int

    def __len__(self) -> int:
        return len(self.books)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        book_example = self.books.iloc[index]
        image_example = self.images.loc[book_example.isbn]

        description = [
            book_example.title,
            book_example.author,
            book_example.publisher,
            book_example.category,
        ]
        description = f" {self.tokenizer.sep_token} ".join(description)

        batch = self.tokenizer(description, truncation=True, max_length=self.max_length)
        batch["labels"] = list(map(int, image_example.tokens.split()))
        return batch
