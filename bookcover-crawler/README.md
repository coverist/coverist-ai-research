## BookCover Crawler

## Introduction
This project contains the codes for crawling book informations and their book-cover images from [교보문고](http://www.kyobobook.co.kr/index.laf?OV_REFFER=https://www.google.com/).

## Requirements
This project requires the below libraries.
* beautifulsoup4
* pandas
* Pillow
* requests
* tqdm

You can easily install them by using the below command:
```bash
$ pip install -r requirements.txt
```

## Getting started
First of all, you need to choose the range of released dates of the books you want to download. The crawler will collect the books which is published in the given range. In this section, we will set the range to `202001 ~ 202012`.
```bash
$ python src/main.py --start-date=202001 --end-date=202012
```
After running the above example command, check if `kyobobook-dataset.csv` file and `images` directory are created. The book-cover images are downloaded and saved to the `images` directory. If you want to modify the name of the output file and the directory, check out `--output-csv` and `--output-image-dir` options.

More detailed options are in below. You can also see this by using `--help` option.
```
usage: main.py [-h] [--start-date START_DATE] [--end-date END_DATE] [--output-csv OUTPUT_CSV]
               [--output-image-dir OUTPUT_IMAGE_DIR] [--num-cores NUM_CORES]

optional arguments:
  -h, --help            show this help message and exit
  --start-date START_DATE
  --end-date END_DATE
  --output-csv OUTPUT_CSV
  --output-image-dir OUTPUT_IMAGE_DIR
  --num-cores NUM_CORES
```

## Results

*Note: keywords are sorted by the similarities between books and corresponding keywords. If you want to use only top-k keywords (truncate or limit the maximum keywords), it would be better to use the first k keywords.*

*Note: there are some books without keywords. Statistically, about 30% of the books do not have their keywords. Consider excluding no-keyword-books because there is a possibility that they are abnormal.*

You can see the crawled 128k book informations in [this file](./kyobobook-dataset.csv). Instead, you can download the entire dataset which consists of the book information file and their book-cover images from [this link](https://drive.google.com/file/d/1HIY32G-UBZzzYzHp1y-_1zwQp7CHO1CK/view?usp=sharing).

Examples of the book information are like this:

|title|author|publisher|published_date|class|barcode|category|keywords|
|--|--|--|--|--|--|--|--|
|프로크리에이트로 시작하는 아이패드 드로잉|수지(허수정)|책밥 |2020-02-03|332323|9791196845391|컴퓨터/IT > 멀티미디어 > 스마트폰/태블릿|드로잉, 애니메이션, 그림, 스케치, 브러시, 캐릭터 그리기, 일러스트레이터, 내기, 리핑 마스크, 투시|
|양준일 Maybe|양준일|모비딕북스 |2020-02-14|030701|9791196601911|시/에세이 > 나라별 에세이 > 한국에세이|한국에세이, 소환, 자전적에세이, 서빙, 가수, 잠언, 연예인, 만개, 텔러, 좌절|
|딸에게 보내는 심리학 편지(10만 부 기념 스페셜 에디션)|한성희|메이븐 |2020-01-28|050301|9791190538015|인문 > 심리학 > 교양심리|감정, 자존감, 교양심리, 인생, 사랑, 니체, 이기주의자, 슈퍼 우먼, 삶의지혜, 개인주의|
|노워리 상담넷: 불안을 주세요, 안심을 드립니다|사교육걱정없는세상 노워리 상담넷|우리학교 |2020-01-17|070501|9791190337236|가정/육아 > 자녀교육 > 자녀교육일반서|영어 학습, 초등, 자녀 교육, 아이, 공부 습관, 부모, 독서 습관, 초등학교, 제안, 스마트폰|
|아메리카노 엑소더스. 7|박지은|소란북스 |2020-02-03|472409|9791189544096|만화 > 웹툰/카툰에세이 > SF/판타지|판타지만화, 단행본, 웹툰, 황혼, 알트, 마법사, 작화, 마법판타지, 음모, 토요|
|...|...|...|...|...|...|...|...|