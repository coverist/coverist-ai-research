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
            "title": title,
            "author": author,
            "publisher": publisher,
            "published_date": pd.to_datetime("/".join(published_date)),
            "class": int(link_class),
            "barcode": int(barcode),
        }
        book_info_list.append(book_info)

    return book_info_list


def get_book_keywords_by_barcodes(barcodes: list[int]) -> list[list[str]]:

    keyword_dict = {}
    for offset in range(0, len(barcodes), constants.CHUNK_SIZE_FOR_KEYWORD_API):
        chunk = barcodes[offset : offset + constants.CHUNK_SIZE_FOR_KEYWORD_API]
        key_list = ",".join(map(str, chunk))

        url = constants.URL_BOOK_KEYWORD_API.format(keyList=key_list)
        with requests.get(url) as resp:
            result = resp.json()

        keyword_dict = keyword_dict | {
            int(barcode): [keyword["itemId"] for keyword in keywords]
            for barcode, keywords in result["groupedResults"].items()
        }

    return [
        keyword_dict[barcode] if barcode in keyword_dict else [] for barcode in barcodes
    ]
    """

    url = constants.URL_BOOK_KEYWORD_API.format(keyList=",".join(map(str, barcodes)))
    with requests.get(url) as resp:
        result = resp.json()

    keywords = {
        int(barcode): [keyword["itemId"] for keyword in keywords]
        for barcode, keywords in result["groupedResults"].items()
    }
    return [keywords[barcode] if barcode in keywords else [] for barcode in barcodes]
    """


def get_book_cover_image(barcode: int) -> Image:
    url = constants.URL_BOOK_COVER_XLARGE.format(
        barcodeSplit=str(barcode)[-3:], barcode=barcode
    )
    with requests.get(url, stream=True) as resp:
        return Image.open(resp.raw)
