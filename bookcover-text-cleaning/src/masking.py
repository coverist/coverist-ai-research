import easyocr
import numpy as np


class BookCoverMask:
    def __init__(self):
        self.reader = easyocr.Reader(["ko", "en"])

    def generate_masks(
        self, batch_images: list[np.ndarray]
    ) -> list[list[list[list[int]]]]:
        n_height, n_width, _ = np.mean([img.shape for img in batch_images], dim=0)

        batch_merged_masks = [[] for _ in batch_images]
        for low_text in np.linspace(0.3, 0.6, 4):
            batch_results = self.reader.readtext_batched(
                batch_images, n_width, n_height, low_text=low_text
            )
            for i, result in enumerate(batch_results):
                for bbox, text, confidence in result:
                    batch_merged_masks[i].append([[int(x), int(y)] for x, y in bbox])

        return batch_merged_masks
