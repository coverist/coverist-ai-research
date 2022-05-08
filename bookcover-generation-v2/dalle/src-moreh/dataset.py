from dataclasses import dataclass
from typing import Dict, List

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

    def decode_alphabet_word_sequence(self, text: str) -> List[int]:
        return [
            sum((ord(letter) - 65) * 26 ** i for i, letter in enumerate(word))
            for word in text.split()
        ]

    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        book_example = self.books.iloc[index]
        image_example = self.images.loc[book_example.isbn]

        description = [
            "제목: " + book_example.title,
            "저자: " + book_example.author,
            "출판사: " + book_example.publisher,
            "카테고리: " + book_example.category,
        ]
        description = " ___ ".join(description)

        batch = self.tokenizer(description, truncation=True, max_length=self.max_length)
        batch["labels"] = self.decode_alphabet_word_sequence(image_example.tokens)
        return batch
