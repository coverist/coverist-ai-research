import easyocr
import numpy as np
from easyocr.utils import reformat_input_batched


class BookCoverMask:
    def __init__(
        self, scaling_factor: float = 1.0, lower_bounds: list = [0.3, 0.4, 0.5, 0.6]
    ):
        self.reader = easyocr.Reader(["ko", "en"], recognizer=False)
        self.scaling_factor = scaling_factor
        self.lower_bounds = lower_bounds

    def _generate_masks_with_lower_bound(
        self, batch_images: list[np.ndarray], lower_bound: float = 0.4
    ) -> list[list[list[list[int]]]]:
        height, width, _ = np.mean([img.shape for img in batch_images], axis=0)
        height, width = height * self.scaling_factor, width * self.scaling_factor

        batch_images, _ = reformat_input_batched(batch_images, int(width), int(height))
        batch_horizontal_list, batch_free_list = self.reader.detect(
            batch_images,
            low_text=lower_bound,
            reformat=False,
        )

        bbox_list = [[] for _ in batch_images]
        for i, horizontal_list in enumerate(batch_horizontal_list):
            if not horizontal_list:
                continue
            for x_min, x_max, y_min, y_max in horizontal_list:
                x_min, x_max = x_min / width, x_max / width
                y_min, y_max = y_min / height, y_max / height
                bbox_list[i].append(
                    [[x_min, y_min], [x_min, y_max], [x_max, y_max], [x_max, y_min]]
                )
        for i, free_list in enumerate(batch_free_list):
            if not free_list:
                continue
            for bbox in free_list:
                for point in bbox:
                    point[0] = point[0] / width
                    point[1] = point[1] / height
                bbox_list[i].append(bbox)
        return bbox_list

    def generate_masks(
        self, batch_images: list[np.ndarray]
    ) -> list[list[list[list[int]]]]:
        merged_text_masks = [[] for _ in batch_images]
        for lower_bound in self.lower_bounds:
            batched_masks = self._generate_masks_with_lower_bound(
                batch_images, lower_bound
            )
            for i, text_mask in enumerate(batched_masks):
                merged_text_masks[i] += text_mask
        return merged_text_masks
