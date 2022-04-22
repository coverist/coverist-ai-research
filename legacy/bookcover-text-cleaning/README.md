# BookCover Text Cleaning

## Introduction
This project contains the code for cleaning the texts (e.g. title, authors, publishers, etc.) from the book-cover images.

## Requirements
This project requires the below libraries.
* numpy
* pandas
* easyocr
* opencv-python-headless==4.5.4.60

You can easily install them by using the below command:
```bash
$ pip install -r requirements.txt
```

## Usage
This project will detect the text regions and extract the bounding boxes from the given book-cover images. You can crawl the bookcover images from [this project](../bookcover-crawler). You should specify both dataset csv file and image folder path. Of course the detecting process is executed on GPU, you may adjust the batch size. Since larger images show better detection results, you can change the upscaling factor by `--scaling-factor` option. Note that the larger images also consume larger GPU memory and it can occur out-of-memory.

More detailed options are in below. You can see this by using `--help` option.
```
usage: main.py [-h] [--output-csv OUTPUT_CSV] [--dataset-path DATASET_PATH]
               [--image-dir IMAGE_DIR] [--lower-bounds LOWER_BOUNDS [LOWER_BOUNDS ...]] 
               [--batch-size BATCH_SIZE] [--scaling-factor SCALING_FACTOR]

optional arguments:
  -h, --help            show this help message and exit
  --output-csv OUTPUT_CSV
  --dataset-path DATASET_PATH
  --image-dir IMAGE_DIR
  --lower-bounds LOWER_BOUNDS [LOWER_BOUNDS ...]
  --batch-size BATCH_SIZE
  --scaling-factor SCALING_FACTOR
```