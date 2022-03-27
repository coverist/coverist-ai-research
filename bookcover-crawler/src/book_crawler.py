from __future__ import annotations

import os
from typing import Any, Optional

from requests import Session

import request_api


class BookCrawler:
    def __new__(cls, *args, **kwargs) -> BookCrawler:
        if not hasattr(cls, "instance"):
            cls.instance = object.__new__(cls)
            cls.instance.__init__(*args, **kwargs)
        return cls.instance

    def __init__(
        self,
        page_size: Optional[int] = None,
        category_dict: Optional[dict[str, str]] = None,
        keyword_chunk_size: Optional[int] = None,
        output_image_dir: Optional[str] = None,
    ):
        if not getattr(BookCrawler, "initialized", False):
            BookCrawler.initialized = True

            self.sess = Session()
            self.page_size = page_size
            self.category_dict = category_dict
            self.keyword_chunk_size = keyword_chunk_size
            self.output_image_dir = output_image_dir

    def create_request_pages(
        self, category: str
    ) -> tuple[int, list[tuple[str, int, int]]]:
        num_books = request_api.get_number_of_books(self.sess, category)
        num_pages = (num_books + self.page_size - 1) // self.page_size
        return num_books, [(category, self.page_size, i + 1) for i in range(num_pages)]

    def get_book_keywords_with_chunking(self, isbn_list: list[str]) -> list[list[str]]:
        keywords_list = []
        for i in range(0, len(isbn_list), self.keyword_chunk_size):
            chunk = isbn_list[i : i + self.keyword_chunk_size]
            keywords_list.extend(request_api.get_book_keywords(self.sess, chunk))
        return keywords_list

    def download_cover_image(self, book: dict[str, Any]):
        if not book["with_cover"]:
            return

        basedir = os.path.join(self.output_image_dir, *book["isbn"][-3:])
        filename = os.path.join(basedir, f"{book['isbn']}.jpg")

        if os.path.exists(filename):
            return
        os.makedirs(basedir, exist_ok=True)

        try:
            image = request_api.get_book_cover_image(self.sess, book["isbn"])
            image.save(os.path.join(basedir, f"{book['isbn']}.jpg"))

            book["cover_aspect_ratio"] = image.width / image.height
        except Exception:
            book["with_cover"] = False
            return

    def collect_and_download(
        self, request_page: tuple[str, int, int]
    ) -> list[dict[str, Any]]:
        try:
            book_list = request_api.get_book_list(self.sess, *request_page)
            book_list = [book for book in book_list if book["minimal_age"] < 19]

            for book in book_list:
                book["linkclass"] = request_page[0]
                book["category"] = self.category_dict[request_page[0]]

            keywords_list = self.get_book_keywords_with_chunking(
                [book["isbn"] for book in book_list]
            )
            for book, keywords in zip(book_list, keywords_list):
                book["keywords"] = ",".join(keywords)

            for book in book_list:
                self.download_cover_image(book)
            return book_list
        except Exception:
            return []

    @staticmethod
    def create_request_pages_static(
        category: str,
    ) -> tuple[int, list[tuple[str, int, int]]]:
        return BookCrawler().create_request_pages(category)

    @staticmethod
    def collect_and_download_static(
        request_page: tuple[str, int, int]
    ) -> list[dict[str, Any]]:
        return BookCrawler().collect_and_download(request_page)
