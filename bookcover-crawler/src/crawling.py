from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup
from PIL import Image

import constants


def get_book_list_by_page_and_date(
    page: int, year: int, month: int, week: int
) -> list[dict[str, Any]]:
    url = constants.URL_BOOK_LIST_BY_DATE_AND_PAGE.format(
        pageNumber=page, year=year, month=month, week=week
    )
    with requests.post(url, data={"targetPage": page}) as resp:
        soup = BeautifulSoup(resp.text, "html.parser")

    book_info_list = []
    for book in soup.select(constants.CSS_SELECTOR_BOOK_TABLE_ROW):
        try:
            title_elem = book.select_one(constants.CSS_SELECTOR_BOOK_TITLE)
            title, detail_page_link = title_elem.text, title_elem["href"]

            description = book.select_one(constants.CSS_SELECTOR_BOOK_DESC).text
            description = " ".join(description.split())

            match = constants.REGEX_LINKCLASS_BARCODE_FROM_HREF.match(detail_page_link)
            link_class, barcode = match.groups()

            if not link_class.strip() or not barcode.strip():
                continue

            match = constants.REGEX_AUTHOR_PUBLISHER_DATE_FROM_DESC.match(description)
            author, publisher, *published_date = match.groups()

            book_info = {
                "title": title.strip(),
                "author": author.strip(),
                "publisher": publisher.strip(),
                "published_date": pd.to_datetime("/".join(published_date)),
                "class": link_class.strip(),
                "barcode": barcode.strip(),
            }
            book_info_list.append(book_info)
        except Exception:
            print("[*] error occurred. skip 1 book...")

    return book_info_list


def get_book_keywords_by_barcodes(barcodes: list[str]) -> list[list[str]]:
    keyword_dict = {}
    for offset in range(0, len(barcodes), constants.CHUNK_SIZE_FOR_KEYWORD_API):
        chunk = barcodes[offset : offset + constants.CHUNK_SIZE_FOR_KEYWORD_API]
        url = constants.URL_BOOK_KEYWORD_API.format(keyList=",".join(chunk))

        with requests.get(url) as resp:
            result = resp.json()

        keyword_dict = keyword_dict | {
            barcode: [keyword["itemId"] for keyword in keywords]
            for barcode, keywords in result["groupedResults"].items()
        }

    return [
        keyword_dict[barcode] if barcode in keyword_dict else [] for barcode in barcodes
    ]


def get_book_cover_image(barcode: str) -> Image:
    url = constants.URL_BOOK_COVER_XLARGE.format(
        barcodeSplit=barcode[-3:], barcode=barcode
    )
    with requests.get(url, stream=True) as resp:
        return Image.open(resp.raw)


def get_category_from_linkclass(linkclass: str) -> list[str]:
    url = constants.URL_CATEGORY_FROM_LINKCLASS.format(linkClass=linkclass)
    with requests.get(url) as resp:
        soup = BeautifulSoup(resp.text, "html.parser")
    return [
        loc.text.strip()
        for loc in soup.select(constants.CSS_SELECTOR_CATEGORY_LOCATION)[1:]
    ]
