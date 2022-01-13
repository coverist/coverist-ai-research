import re

import requests

URL = (
    "http://www.kyobobook.co.kr/newproduct/newTopicKorList.laf"
    "?pageNumber={pageNumber}&perPage=50&mallGb=KOR&linkClass=&sortColumn=near_date"
    "&newYmw={year:04d}{month:02d}{week:1d}"
)
tmp = (
    "http://www.kyobobook.co.kr/product/detailViewKor.laf"
    "?mallGb=KOR&ejkGb=KOR&linkClass={linkClass}&barcode={barcode}"
)

with requests.get(URL.format(pageNumber=1, year=2020, month=1, week=1)) as resp:
    print(resp.text)
