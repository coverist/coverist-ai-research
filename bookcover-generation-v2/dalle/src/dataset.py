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

    def create_token_type_ids(self, input_ids: list[int]) -> list[int]:
        token_type_ids = [0]
        for i in input_ids[1:]:
            if i == self.tokenizer.sep_token_id:
                token_type_ids.append(token_type_ids[-1] + 1)
            else:
                token_type_ids.append(token_type_ids[-1])
        return [0] + token_type_ids[:-1]

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
        batch["token_type_ids"] = self.create_token_type_ids(batch["input_ids"])
        batch["labels"] = list(map(int, image_example.tokens.split()))
        return batch
