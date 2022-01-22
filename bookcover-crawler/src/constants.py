import re

URL_BOOK_LIST_BY_DATE_AND_PAGE = (
    "http://www.kyobobook.co.kr/newproduct/newTopicKorList.laf"
    "?pageNumber={pageNumber}&perPage=50&mallGb=KOR"
    "&newYmw={year:04d}{month:02d}{week:1d}"
    "&yyyy={year:04d}&mm={month:02d}&week={week:1d}"
)

URL_BOOK_KEYWORD_API = "http://api.eigene.io/rec/kyobo002?format=jsonp&key={keyList}"
URL_BOOK_COVER_XLARGE = (
    "http://image.kyobobook.co.kr/images/book/xlarge/{barcodeSplit}/x{barcode}.jpg"
)
URL_CATEGORY_FROM_LINKCLASS = (
    "http://www.kyobobook.co.kr/categoryRenewal/categoryMain.laf?"
    "linkClass={linkClass}&mallGb=KOR&orderClick=JAR"
)

CSS_SELECTOR_BOOK_TABLE_ROW = "dl.book_title"
CSS_SELECTOR_BOOK_TITLE = "dt strong a"
CSS_SELECTOR_BOOK_DESC = "dd"
CSS_SELECTOR_CATEGORY_LOCATION = "p.location"

REGEX_LINKCLASS_BARCODE_FROM_HREF = re.compile(
    r"javascript:goDetailProductNotAge\('KOR','\s*(\d*)\s*','\s*(\d*)\s*',.*\)"
)
REGEX_AUTHOR_PUBLISHER_DATE_FROM_DESC = re.compile(
    r"^([^|]+) 지음 \|(?:[^|]*\|)?\s*([^|]+)\s*\| (\d+)\D*(\d+)\D*(\d+).*"
)

CHUNK_SIZE_FOR_KEYWORD_API = 20
