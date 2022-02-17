import argparse
import os

import cv2
import numpy as np
import pandas as pd
import tqdm

from masking import BookCoverMask


def read_image_by_barcode(barcode: str, base_dir: str) -> np.ndarray:
    image = cv2.imread(os.path.join(base_dir, *barcode[-3:], f"{barcode}.jpg"))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


def main(args: argparse.Namespace):
    mask = BookCoverMask(args.lower_bounds)

    dataset = pd.read_csv(args.dataset_path, dtype={"barcode": str})
    barcodes = dataset.barcode.tolist()

    result = []
    for i in tqdm.trange(0, len(barcodes), args.batch_size):
        batch_barcodes = barcodes[i : i + args.batch_size]
        batch_images = [
            read_image_by_barcode(barcode, args.image_dir) for barcode in batch_barcodes
        ]
        batch_masks = mask.generate_masks(batch_images)

        for barcode, image, bbox in zip(batch_barcodes, batch_images, batch_masks):
            bbox_text = "\n".join(
                [" ".join([f"{x} {y}" for x, y in bb]) for bb in bbox]
            )
            row_item = {
                "barcode": barcode,
                "width": image.shape[1],
                "height": image.shape[0],
                "bbox": bbox_text,
            }
            result.append(row_item)

    pd.DataFrame(result).to_csv(args.output_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-csv", default="bookcover-text-masking.csv")
    parser.add_argument("--dataset-path", default="../resources/kyobobook-dataset.csv")
    parser.add_argument("--image-dir", default="../resources/images")
    parser.add_argument(
        "--lower-bounds", nargs="+", type=int, default=[0.3, 0.4, 0.5, 0.6]
    )
    parser.add_argument("--batch-size", type=int, default=32)
    main(parser.parse_args())
