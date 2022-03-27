from PIL import Image
from requests import Session

URL_REQUEST_BOOK_LIST = (
    "http://mobile.kyobobook.co.kr/search/bycategoryAjax/KOR/{}/Date?size={}&offset={}"
)
URL_REQUEST_KEYWORDS = "http://api.eigene.io/rec/kyobo002?format=jsonp&key={}"
URL_REQUEST_COVER_IMAGE = "http://image.kyobobook.co.kr/images/book/xlarge/{}/x{}.jpg"


def create_book_info_from_result(result: dict[str, object]) -> dict[str, object]:
    return {
        "title": result["title"],
        "description": result["subTitle"],
        "author": result["authorName"],
        "translator": result["translatorName"],
        "publisher": result["publisherName"],
        "published_date": result["publishingDay"],
        "isbn": result["barcode"],
        "linkclass": "",
        "category": "",
        "keywords": "",
        "with_cover": result["withCover"] == "true",
        "cover_aspect_ratio": 0.0,
        "content_rating": result["readerContentGradeAverage"],
        "design_rating": result["readerDesignGradeAverage"],
        "minimal_age": result["minimalAge"],
        "price": result["listPrice"],
    }


def get_number_of_books(sess: Session, linkclass: str) -> int:
    with sess.get(URL_REQUEST_BOOK_LIST.format(linkclass, 1, 1)) as resp:
        return resp.json()["totalCount"]


def get_book_list(
    sess: Session, linkclass: str, size: int, offset: int
) -> list[dict[str, object]]:
    with sess.get(URL_REQUEST_BOOK_LIST.format(linkclass, size, offset)) as resp:
        book_list = list(map(create_book_info_from_result, resp.json()["resultList"]))
    for book in book_list:
        book["linkclass"] = linkclass
    return book_list


def get_book_keywords(sess: Session, isbn_list: list[str]) -> list[list[str]]:
    with sess.get(URL_REQUEST_KEYWORDS.format(",".join(isbn_list))) as resp:
        results = resp.json()["groupedResults"]
    return [
        [keyword["itemId"] for keyword in results[isbn]] if isbn in results else []
        for isbn in isbn_list
    ]


def get_book_cover_image(sess: Session, isbn: str) -> Image.Image:
    with sess.get(URL_REQUEST_COVER_IMAGE.format(isbn[-3:], isbn), stream=True) as resp:
        return Image.open(resp.raw)
