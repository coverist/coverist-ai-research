import argparse
import os

import imagesize
import pandas as pd


def get_aspect_ratio(filename: str) -> float:
    width, height = imagesize.get(filename)
    return width / height


def main(args: argparse.Namespace):
    dataset = pd.read_csv(args.dataset_path, dtype={"barcode": "str"})
    print(f"[*] total number of books: {len(dataset)}")

    # Remove the book-cover images of which aspect ratios are abnormal.
    aspect_ratio_list = [
        get_aspect_ratio(os.path.join(args.image_dir, *barcode[-3:], f"{barcode}.jpg"))
        for barcode in dataset.barcode
    ]
    aspect_ratio_mask = [
        args.min_aspect_ratio <= ratio <= args.max_aspect_ratio
        for ratio in aspect_ratio_list
    ]
    dataset = dataset[aspect_ratio_mask]
    print(f"[*] total number of valid-aspect-ratio books: {len(dataset)}")

    # Remove categories which have insufficient amount of books.
    counts = dataset.category.value_counts()
    dataset = pd.merge(dataset, counts, left_on="category", right_index=True)
    dataset = dataset[dataset.category_y >= args.min_images_per_category]
    dataset = dataset.drop("category_y", axis=1)
    print(f"[*] total number of sufficient-category books: {len(dataset)}")

    dataset.to_csv(args.output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", default="kyobobook-dataset.csv")
    parser.add_argument("--image-dir", default="kyobobook-images")
    parser.add_argument("--output-csv", default="kyobobook-dataset-filtered.csv")
    parser.add_argument("--min-aspect-ratio", type=float, default=0.6)
    parser.add_argument("--max-aspect-ratio", type=float, default=0.8)
    parser.add_argument("--min_images_per_category", type=int, default=50)
    main(parser.parse_args())
