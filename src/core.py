from collections import defaultdict
from typing import Any

import cv2
import numpy as np

import constants as constants
from logger import logger
from threshholdCalculator import ThresholdCalculator
from utils.image import CLAHE_HELPER, ImageUtils


class ImageInstanceOps:
    """Class to hold fine-tuned utilities for a group of images. One instance for each processing directory."""

    save_img_list: Any = defaultdict(list)

    def __init__(self, tuning_config):
        super().__init__()
        self.tuning_config = tuning_config
        self.save_image_level = tuning_config.outputs.save_image_level

    def apply_preprocessors(self, file_path, in_omr, template):
        tuning_config = self.tuning_config

        in_omr = ImageUtils.resize_util(
            in_omr,
            tuning_config.dimensions.processing_width,
            tuning_config.dimensions.processing_height,
        )

        for pre_processor in template.pre_processors:
            in_omr = pre_processor.apply_filter(in_omr, file_path)
        return in_omr

    def apply_auto_align(self, config, template, image):
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        image = CLAHE_HELPER.apply(image)

        # Adjust gamma for the image
        image = ImageUtils.adjust_gamma(image, config.threshold_params.GAMMA_LOW)

        # Apply threshold truncation
        _, image = cv2.threshold(image, 220, 220, cv2.THRESH_TRUNC)
        image = ImageUtils.normalize_util(image)

        # Create a vertical structuring element for morphological operations
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 10))

        # Apply morphological open operation
        image_vertical = cv2.morphologyEx(
            image, cv2.MORPH_OPEN, vertical_kernel, iterations=3
        )

        # Apply threshold truncation to the vertical image
        _, image_vertical = cv2.threshold(image_vertical, 200, 200, cv2.THRESH_TRUNC)
        image_vertical = 255 - ImageUtils.normalize_util(image_vertical)

        # Apply binary thresholding
        morph_threshold = 60
        _, image_vertical = cv2.threshold(
            image_vertical, morph_threshold, 255, cv2.THRESH_BINARY
        )

        # Erode the image
        image_vertical = cv2.erode(
            image_vertical, np.ones((5, 5), np.uint8), iterations=2
        )

        # Iterate through each field block in the template
        for field_block in template.field_blocks:
            origin, dimensions = field_block.origin, field_block.dimensions

            # Retrieve alignment parameters from the configuration
            match_col, max_steps, align_stride, thickness = (
                config.alignment_params.get(param)
                for param in ["match_col", "max_steps", "stride", "thickness"]
            )

            shift, steps = 0, 0
            while steps < max_steps:
                # Calculate the mean pixel value on the left side of the field block
                left_mean = np.mean(
                    image_vertical[
                        origin[1] : origin[1] + dimensions[1],
                        origin[0]
                        + shift
                        - thickness : origin[0]
                        + shift
                        + match_col
                        - thickness,
                    ]
                )

                # Calculate the mean pixel value on the right side of the field block
                right_mean = np.mean(
                    image_vertical[
                        origin[1] : origin[1] + dimensions[1],
                        origin[0]
                        + shift
                        + dimensions[0]
                        - match_col
                        + thickness : origin[0]
                        + shift
                        + dimensions[0]
                        + thickness,
                    ]
                )

                # Determine shift direction based on mean values
                left_shift, right_shift = left_mean > 100, right_mean > 100
                if left_shift:
                    if right_shift:
                        break
                    else:
                        shift -= align_stride
                else:
                    if right_shift:
                        shift += align_stride
                    else:
                        break
                steps += 1

            # Assign the calculated shift to the field block
            field_block.shift = shift

        return template.field_blocks

    def read_omr_response(self, template, image):
        config = self.tuning_config
        auto_align = True
        threshhold_calculator = ThresholdCalculator(config)

        try:
            img = image.copy()
            img = cv2.flip(img, 1)
            img = ImageUtils.resize_util(
                img, template.page_dimensions[0], template.page_dimensions[1]
            )
            if img.max() > img.min():
                img = ImageUtils.normalize_util(img)

            final_marked = img.copy()

            morph = img.copy()

            omr_response = {}

            if auto_align:
                template.field_blocks = self.apply_auto_align(config, template, morph)

            # Initialize lists to store all average values, all strip values arrays, and all standard deviation values
            all_mean_values = []
            all_strip_values_arrays = []
            all_standard_deviation_values = []
            total_strip_count = 0

            for field_block in template.field_blocks:
                bubble_width, bubble_height = field_block.bubble_dimensions
                field_standard_deviation_values = []

                # Traverse through bubbles in the field block
                for bubble_group in field_block.traverse_bubbles:
                    strip_values = []

                    # Calculate mean values for each bubble point
                    for bubble_point in bubble_group:
                        x = bubble_point.x + field_block.shift
                        y = bubble_point.y
                        rect = [y, y + bubble_height, x, x + bubble_width]
                        mean_value = cv2.mean(
                            img[rect[0] : rect[1], rect[2] : rect[3]]
                        )[0]
                        strip_values.append(mean_value)

                    # Calculate the standard deviation of the strip values and round to 2 decimal places
                    field_standard_deviation_values.append(
                        round(np.std(strip_values), 2)
                    )

                    # Append the strip values to the overall arrays
                    all_strip_values_arrays.append(strip_values)
                    all_mean_values.extend(strip_values)
                    total_strip_count += 1

                # Extend the overall standard deviation values with the field's standard deviation values
                all_standard_deviation_values.extend(field_standard_deviation_values)

            global_std_thresh, _, _ = threshhold_calculator.get_global_threshold(
                all_standard_deviation_values
            )

            global_thr, _, _ = threshhold_calculator.get_global_threshold(
                all_mean_values, looseness=4
            )

            logger.info(
                f"Thresholding:\tglobal_thr: {round(global_thr, 2)} \tglobal_std_THR: {round(global_std_thresh, 2)}\t{'(Looks like a Xeroxed OMR)' if (global_thr == 255) else ''}"
            )
            # Initialize variables for OMR threshold average, total strip count, and total box count
            omr_threshold_avg = 0
            total_strip_count = 0
            total_box_count = 0

            # Convert the final marked image to BGR color space
            final_marked = cv2.cvtColor(final_marked, cv2.COLOR_GRAY2BGR)

            # Iterate through each field block in the template
            for field_block in template.field_blocks:
                block_strip_count = 1
                bubble_width, bubble_height = field_block.bubble_dimensions
                block_key = field_block.name[:3]

                question_index = 0
                for bubble_group in field_block.traverse_bubbles:

                    # Determine if there are no outliers based on standard deviation
                    no_outliers = (
                        all_standard_deviation_values[total_strip_count]
                        < global_std_thresh
                    )

                    # Get the local threshold for the current strip of bubbles
                    strip_threshold = threshhold_calculator.get_local_threshold(
                        all_strip_values_arrays[total_strip_count],
                        global_thr,
                        no_outliers,
                    )

                    omr_threshold_avg += strip_threshold

                    detected_bubbles = []
                    for bubble in bubble_group:
                        is_bubble_marked = (
                            strip_threshold > all_mean_values[total_box_count]
                        )
                        total_box_count += 1

                        if is_bubble_marked:
                            detected_bubbles.append(bubble)
                            x = bubble.x + field_block.shift
                            y = bubble.y

                            # Draw rectangles based on marking status
                            if field_block.correct_answers is None:
                                color = (
                                    0,
                                    0,
                                    255,
                                )
                            elif (
                                field_block.correct_answers[question_index]
                                == bubble.field_value
                            ):
                                color = (0, 255, 0)
                            else:
                                color = (255, 0, 0)

                            cv2.rectangle(
                                final_marked,
                                (
                                    int(x + bubble_width / 12),
                                    int(y + bubble_height / 12),
                                ),
                                (
                                    int(x + bubble_width - bubble_width / 12),
                                    int(y + bubble_height - bubble_height / 12),
                                ),
                                color,
                                2,
                            )

                    # Record detected bubbles in the OMR response
                    for bubble in detected_bubbles:
                        field_label = bubble.field_label
                        field_value = bubble.field_value
                        omr_response[field_label] = field_value

                    # Handle case where no bubbles are detected
                    if len(detected_bubbles) == 0:
                        field_label = bubble_group[0].field_label
                        omr_response[field_label] = field_block.empty_val

                    block_strip_count += 1
                    total_strip_count += 1
                    question_index += 1

            # Calculate the average OMR threshold and flip the final marked image
            omr_threshold_avg /= total_strip_count
            omr_threshold_avg = round(omr_threshold_avg, 2)

            final_marked = cv2.flip(final_marked, 2)
            return (omr_response, final_marked)

        except Exception as e:
            raise e

    @staticmethod
    def draw_template_layout(img, template, shifted=True, draw_qvals=False, border=-1):
        img = ImageUtils.resize_util(
            img, template.page_dimensions[0], template.page_dimensions[1]
        )
        final_align = img.copy()
        for field_block in template.field_blocks:
            s, d = field_block.origin, field_block.dimensions
            box_w, box_h = field_block.bubble_dimensions
            shift = field_block.shift
            if shifted:
                cv2.rectangle(
                    final_align,
                    (s[0] + shift, s[1]),
                    (s[0] + shift + d[0], s[1] + d[1]),
                    constants.CLR_BLACK,
                    3,
                )
            else:
                cv2.rectangle(
                    final_align,
                    (s[0], s[1]),
                    (s[0] + d[0], s[1] + d[1]),
                    constants.CLR_BLACK,
                    3,
                )
            for field_block_bubbles in field_block.traverse_bubbles:
                for bubble_point in field_block_bubbles:
                    x, y = (
                        (bubble_point.x + field_block.shift, bubble_point.y)
                        if shifted
                        else (bubble_point.x, bubble_point.y)
                    )
                    cv2.rectangle(
                        final_align,
                        (int(x + box_w / 10), int(y + box_h / 10)),
                        (int(x + box_w - box_w / 10), int(y + box_h - box_h / 10)),
                        constants.CLR_GRAY,
                        border,
                    )
                    if draw_qvals:
                        rect = [y, y + box_h, x, x + box_w]
                        cv2.putText(
                            final_align,
                            f"{int(cv2.mean(img[rect[0] : rect[1], rect[2] : rect[3]])[0])}",
                            (rect[2] + 2, rect[0] + (box_h * 2) // 3),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            constants.CLR_BLACK,
                            2,
                        )
            if shifted:
                text_in_px = cv2.getTextSize(
                    field_block.name, cv2.FONT_HERSHEY_SIMPLEX, constants.TEXT_SIZE, 4
                )
                cv2.putText(
                    final_align,
                    field_block.name,
                    (int(s[0] + d[0] - text_in_px[0][0]), int(s[1] - text_in_px[0][1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    constants.TEXT_SIZE,
                    constants.CLR_BLACK,
                    4,
                )
        return final_align
