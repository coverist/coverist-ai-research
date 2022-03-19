import easyocr
import numpy as np


class BookCoverMask:
    def __init__(
        self, scaling_factor: float = 1.0, lower_bounds: list = [0.3, 0.4, 0.5, 0.6]
    ):
        self.reader = easyocr.Reader(["ko", "en"])
        self.scaling_factor = scaling_factor
        self.lower_bounds = lower_bounds

    def _generate_masks_with_lower_bound(
        self, batch_images: list[np.ndarray], lower_bound: float = 0.4
    ) -> list[list[list[list[int]]]]:
        height, width, _ = np.mean([img.shape for img in batch_images], axis=0)
        height, width = height * self.scaling_factor, width * self.scaling_factor
        batch_results = self.reader.readtext_batched(
            batch_images, int(width), int(height), low_text=lower_bound
        )

        bbox_list = []
        for image, result in zip(batch_images, batch_results):
            bbox = [bb for bb, _, _ in result]
            for bb in bbox:
                for point in bb:
                    point[0] = point[0] / width
                    point[1] = point[1] / height
            bbox_list.append(bbox)
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
