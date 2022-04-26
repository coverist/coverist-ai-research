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
            "제목: " + book_row.title,
            "저자: " + book_row.author,
            "출판사: " + book_row.publisher,
            "카테고리: " + book_row.category,
        ]
        book_prompt = " \\ ".join(book_prompt)

        batch = self.tokenizer(book_prompt, truncation=True, max_length=self.max_length)
        batch["labels"] = list(map(int, image_row.tokens.split()))
        return batch
