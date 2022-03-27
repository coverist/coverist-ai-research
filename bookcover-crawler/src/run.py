import argparse
import json
import multiprocessing as mp

import pandas as pd
import tqdm

from book_crawler import BookCrawler


def main(args: argparse.Namespace):
    with open(args.category_file) as fp:
        category_dict = json.load(fp)
    request_pages, total_books = [], 0

    with mp.Pool(
        args.num_cores if args.num_cores != -1 else mp.cpu_count(),
        initializer=BookCrawler,
        initargs=(args.page_size, None, None, None),
    ) as pool:
        for num_books, pages in pool.imap_unordered(
            BookCrawler.create_request_pages_static, category_dict
        ):
            print(f"[*] number of books in [{pages[0][0]}]: {num_books}")
            total_books += num_books
            request_pages.extend(pages)
        print(f"[*] total number of books: {total_books}")

    with mp.Pool(
        args.num_cores if args.num_cores != -1 else mp.cpu_count(),
        initializer=BookCrawler,
        initargs=(None, category_dict, args.keyword_chunk_size, args.output_image_dir),
    ) as pool:
        iterator = tqdm.tqdm(
            pool.imap_unordered(BookCrawler.collect_and_download_static, request_pages),
            total=len(request_pages),
        )
        results = [book for book_list in iterator for book in book_list]

    pd.DataFrame(results).to_json(args.output_filename, orient="records", lines=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category-file", default="category.json")
    parser.add_argument("--output-filename", default="./kyobobook-dataset.jsonl")
    parser.add_argument("--output-image-dir", default="./kyobobook-images")
    parser.add_argument("--page-size", type=int, default=1000)
    parser.add_argument("--num-cores", type=int, default=-1)
    parser.add_argument("--keyword-chunk-size", type=int, default=20)
    main(parser.parse_args())
