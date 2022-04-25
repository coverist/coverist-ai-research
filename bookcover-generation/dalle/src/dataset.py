from dataclasses import dataclass

import pandas as pd
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


@dataclass
class DALLEBookDataset(Dataset):
    tokenizer: PreTrainedTokenizerBase
    books: pd.DataFrame
    images: pd.DataFrame
    max_length: int = 64
    image_vocab_size: int = 8192

    def __len__(self) -> int:
        return len(self.books)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        book_row = self.books.iloc[index]
        image_row = self.images.loc[book_row.isbn]

        book_prompt = [
            book_row.title,
            book_row.author,
            book_row.publisher,
            book_row.category,
        ]
        book_prompt = " @ ".join(book_prompt)
        image_tokens = list(map(int, image_row.tokens.split()))

        batch = self.tokenizer(book_prompt, truncation=True, max_length=self.max_length)
        batch["labels"] = [self.image_vocab_size] + image_tokens
        return batch
