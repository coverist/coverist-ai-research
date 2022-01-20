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
First of all, you need to choose the range of released dates. The crawler will collect the books which is published in the given range. In this section, we will set the range to `202001 ~ 202012`.
```bash
$ python src/main.py --start-date=202001 --end-date=202012
```
After running the above example, check if `kyobobook-dataset.csv` file and `images` directory are created. The book-cover images are downloaded and saved to the `images` directory. If you want to modify the name of output file and directory, check out `--output-csv` and `--output-image-dir` options.

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

*Note: keywords are sorted by the similarity between the book and corresponding keywords. If you want to use only top-k keywords (truncate or limit the maximum keywords), it would be better to use the first k keywords.*

*Note: there are some books without keywords. Statistically, about 20% of the books do not have their keywords. Consider excluding no-keyword-books because there are possibilities that they are abnormal.*

|title|author|publisher|published_date|class|barcode|keywords|
|--|--|--|--|--|--|--|
|박막례시피|박막례|미디어창비|2020-09-15|80317|9791190758185|유튜브, 레시피, 요리, 할머니, 유라, 유튜버, 국물 요리, 집밥, 두부김치, 콩나물 무침|
|남매의 여름밤 각본집|윤단비|플레인아카이브|2020-09-13|23191507|9791190738064|에세이, 시나리오집, 단상|
|세포(Editorial Science: 모두를 위한 과학)|남궁석|에디토리얼|2020-08-27|29051901|9791190254045|단백질, 생물학, 알렉시, 생물이야기, 텔로미어, 주기율표, 유전체, 프리드리히, 염색체, 매드|
|공부란 무엇인가|김영민|어크로스|2020-08-26|50101|9791190030632|인문교양서, 질문, 에세이, 평생 공부, 근육, 배움, 무용, 읽기, 심화, 쓰기|
|하루 10분의 기적 초등 패턴 글쓰기|남낙현|청림라이프|2020-09-09|70513|9791188700677|자녀교육, 국어사전, 감정, 질문, 초등학생 아이, 상상력, 학습교육, 스마트폰 사용, 논리력, 문해력|
|...|...|...|...|...|...|...|