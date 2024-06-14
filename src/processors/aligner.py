import cv2
import numpy as np
from utils.image import CLAHE_HELPER, ImageUtils


class Aligner:
    def __init__(self, config):
        self.config = config

    def apply_auto_align(self, template, image):
        image = self._apply_clahe(image)
        image = self._adjust_gamma(image)
        image = self._apply_threshold_truncation(image)
        image = self._normalize_image(image)

        image_vertical = self._apply_morphological_operations(image)
        image_vertical = self._apply_threshold_truncation(image_vertical)
        image_vertical = self._invert_and_normalize_image(image_vertical)
        image_vertical = self._apply_binary_thresholding(image_vertical)
        image_vertical = self._erode_image(image_vertical)

        return self._align_field_blocks(template, image_vertical)

    def _apply_clahe(self, image):
        return CLAHE_HELPER.apply(image)

    def _adjust_gamma(self, image):
        return ImageUtils.adjust_gamma(image, self.config.threshold_params.GAMMA_LOW)

    def _apply_threshold_truncation(self, image, threshold=220):
        _, image = cv2.threshold(image, threshold, threshold, cv2.THRESH_TRUNC)
        return image

    def _normalize_image(self, image):
        return ImageUtils.normalize_util(image)

    def _apply_morphological_operations(self, image):
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))
        return cv2.morphologyEx(image, cv2.MORPH_OPEN, vertical_kernel, iterations=3)

    def _invert_and_normalize_image(self, image):
        return 255 - self._normalize_image(image)

    def _apply_binary_thresholding(self, image, threshold=60):
        _, image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
        return image

    def _erode_image(self, image):
        return cv2.erode(image, np.ones((5, 5), np.uint8), iterations=2)

    def _align_field_blocks(self, template, image_vertical):
        for field_block in template.field_blocks:
            field_block.shift = self._calculate_shift(field_block, image_vertical)
        return template.field_blocks

    def _calculate_shift(self, field_block, image_vertical):
        origin, dimensions = field_block.origin, field_block.dimensions
        match_col, max_steps, align_stride, thickness = (
            self.config.alignment_params.get(param)
            for param in ["match_col", "max_steps", "stride", "thickness"]
        )

        shift, steps = 0, 0
        while steps < max_steps:
            left_mean = self._calculate_mean(
                image_vertical,
                origin,
                dimensions,
                shift,
                match_col,
                thickness,
                side="left",
            )
            right_mean = self._calculate_mean(
                image_vertical,
                origin,
                dimensions,
                shift,
                match_col,
                thickness,
                side="right",
            )

            left_shift, right_shift = left_mean > 100, right_mean > 100
            if left_shift and not right_shift:
                shift -= align_stride
            elif right_shift and not left_shift:
                shift += align_stride
            else:
                break
            steps += 1

        return shift

    def _calculate_mean(
        self, image, origin, dimensions, shift, match_col, thickness, side
    ):
        y_start, y_end = origin[1], origin[1] + dimensions[1]
        if side == "left":
            x_start, x_end = (
                origin[0] + shift - thickness,
                origin[0] + shift + match_col - thickness,
            )
        else:  # side == 'right'
            x_start, x_end = (
                origin[0] + shift + dimensions[0] - match_col + thickness,
                origin[0] + shift + dimensions[0] + thickness,
            )
        return np.mean(image[y_start:y_end, x_start:x_end])
