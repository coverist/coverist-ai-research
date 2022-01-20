import argparse
import multiprocessing as mp
import os
from typing import Any

import pandas as pd
import tqdm

from crawling import (
    get_book_cover_image,
    get_book_keywords_by_barcodes,
    get_book_list_by_page_and_date,
)


def crawl_book_info(year: int, month: int) -> list[dict[str, Any]]:
    total_book_info_list = []
    for week in range(1, 6):
        for page in range(1, 100):
            book_info_list = get_book_list_by_page_and_date(page, year, month, week)

            if (
                not book_info_list
                or total_book_info_list
                and total_book_info_list[-1]["barcode"] == book_info_list[-1]["barcode"]
            ):
                break

            book_keywords_list = get_book_keywords_by_barcodes(
                [book_info["barcode"] for book_info in book_info_list]
            )
            for i, keywords in enumerate(book_keywords_list):
                book_info_list[i]["keywords"] = ", ".join(keywords)

            total_book_info_list += book_info_list

    return total_book_info_list


def process_fn(date_range: pd.DatetimeIndex, output_image_dir: str, queue: mp.Queue):
    for date in date_range:
        book_info_list = []

        for book_info in crawl_book_info(date.year, date.month):
            try:
                image = get_book_cover_image(book_info["barcode"])
                image_path = os.path.join(
                    output_image_dir,
                    *str(book_info["barcode"])[-3:],
                    str(book_info["barcode"]) + ".jpg",
                )

                os.makedirs(os.path.dirname(image_path), exist_ok=True)
                image.save(image_path)

                book_info_list.append(book_info)
            except Exception:
                pass

        queue.put(book_info_list)
    queue.put(None)


def main(args: argparse.Namespace):
    date_range = pd.date_range(f"{args.start_date}01", f"{args.end_date}01", freq="MS")

    processes, terminated, queue = [], 0, mp.Queue()
    for i in range(args.num_cores):
        p = mp.Process(
            target=process_fn,
            args=(date_range[i :: args.num_cores], args.output_image_dir, queue),
            daemon=True,
        )
        p.start()
        processes.append(p)

    total_book_info_list = []
    with tqdm.trange(len(date_range)) as tbar:
        while terminated < len(processes):
            book_info_list = queue.get()

            if book_info_list is None:
                terminated += 1
                continue

            total_book_info_list += book_info_list
            tbar.update()

    for p in processes:
        p.join()

    pd.DataFrame(total_book_info_list).to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="201001")
    parser.add_argument("--end-date", default="202012")
    parser.add_argument("--output-csv", default="kyobobook-dataset.csv")
    parser.add_argument("--output-image-dir", default="./images/")
    parser.add_argument("--num-cores", type=int, default=mp.cpu_count())
    main(parser.parse_args())
