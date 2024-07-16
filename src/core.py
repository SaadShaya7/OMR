import cv2
import numpy as np

from logger import logger
from processors.aligner import Aligner
from processors.CropOnMarkers import CropOnMarkers
from defaults.config import CONFIG_DEFAULTS
from threshholdCalculator import ThresholdCalculator
from utils.image import ImageUtils


class ImageInstanceOps:
    """Class to hold fine-tuned utilities for a group of images. One instance for each processing directory."""

    def __init__(self):
        super().__init__()
        self.tuning_config = CONFIG_DEFAULTS

    def apply_preprocessors(self, file_path, in_omr):
        tuning_config = self.tuning_config

        in_omr = ImageUtils.resize_util(
            in_omr,
            tuning_config.dimensions.processing_width,
            tuning_config.dimensions.processing_height,
        )

        in_omr = CropOnMarkers().apply_filter(in_omr, file_path)
        return in_omr

    def prepare_image(self, image, template):
        """Prepare the image for processing."""
        img = image.copy()
        img = cv2.flip(img, 1)
        img = ImageUtils.resize_util(
            img, template.page_dimensions[0], template.page_dimensions[1]
        )
        if img.max() > img.min():
            img = ImageUtils.normalize_util(img)
        return img

    def process_field_blocks(
        self,
        final_marked,
        template,
        all_mean_values,
        all_strip_values_arrays,
        global_thr,
        global_std_thresh,
        threshhold_calculator,
        all_standard_deviation_values,
    ):
        """Process each field block in the template."""
        omr_response = {}
        omr_threshold_avg = 0
        total_strip_count = 0
        total_box_count = 0
        multi_marked_fields = []

        final_marked = cv2.cvtColor(final_marked, cv2.COLOR_GRAY2BGR)

        for field_block in template.field_blocks:
            block_strip_count = 1
            question_index = 0
            bubble_width, bubble_height = field_block.bubble_dimensions
            for bubble_group in field_block.traverse_bubbles:

                no_outliers = (
                    all_standard_deviation_values[total_strip_count] < global_std_thresh
                )

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
                    # print(f'{question_index} strip threshhold is {strip_threshold}, marked: {is_bubble_marked}')
                    total_box_count += 1

                    if is_bubble_marked:
                        detected_bubbles.append(bubble)
                        self.mark_bubble(
                            bubble,
                            field_block,
                            final_marked,
                            bubble_width,
                            bubble_height,
                            question_index,
                        )

                for bubble in detected_bubbles:
                    field_label = bubble.field_label
                    field_value = bubble.field_value
                    multi_marked_local = field_label in omr_response
                    omr_response[field_label] = (
                        None if multi_marked_local else field_value
                    )
                    if (
                        field_block.name != "StudentId" and field_block.name != "form"
                    ):  # Disregard form and student id from multi answered question count
                        if (
                            multi_marked_local
                            and field_label not in multi_marked_fields
                        ):
                            multi_marked_fields.append(field_label)

                if len(detected_bubbles) == 0:
                    field_label = bubble_group[0].field_label
                    omr_response[field_label] = field_block.empty_val

                block_strip_count += 1
                total_strip_count += 1
                question_index += 1

        omr_threshold_avg /= total_strip_count
        omr_threshold_avg = round(omr_threshold_avg, 2)

        final_marked = cv2.flip(final_marked, 2)
        multi_marked_count = len(multi_marked_fields)
        return (omr_response, final_marked, multi_marked_count)

    def calculate_mean_values(
        self, img, bubble_group, field_block, bubble_width, bubble_height
    ):
        """Calculate mean values for each bubble point."""
        return [
            cv2.mean(
                img[
                    bubble_point.y : bubble_point.y + bubble_height,
                    bubble_point.x
                    + field_block.shift : bubble_point.x
                    + field_block.shift
                    + bubble_width,
                ]
            )[0]
            for bubble_point in bubble_group
        ]

    def process_bubble_group(
        self,
        img,
        bubble_group,
        field_block,
        bubble_width,
        bubble_height,
        all_mean_values,
        all_standard_deviation_values,
    ):
        """Process a group of bubbles."""
        strip_values = self.calculate_mean_values(
            img, bubble_group, field_block, bubble_width, bubble_height
        )
        all_mean_values.extend(strip_values)
        all_standard_deviation_values.append(round(np.std(strip_values), 2))
        return strip_values

    def mark_bubble(
        self,
        bubble,
        field_block,
        final_marked,
        bubble_width,
        bubble_height,
        question_index,
    ):
        x = bubble.x + field_block.shift
        y = bubble.y

        if field_block.correct_answers == None:
            color = (0, 0, 255)
        elif field_block.correct_answers[question_index] == bubble.field_value:
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

    def crop_name(self, original_image):
        origin = (275, 114)
        width = 248
        height = 60
        end_point = (origin[0] + width, origin[1] + height)
        cropped = original_image[origin[1] : end_point[1], origin[0] : end_point[0]]
        cv2.rectangle(original_image, origin, end_point, (0, 0, 255), 2)
        return cropped

    def read_omr_response(self, template, image):
        config = self.tuning_config
        threshhold_calculator = ThresholdCalculator(config)
        aligner = Aligner(config)

        try:
            img = self.prepare_image(image, template)
            final_marked = img.copy()

            template.field_blocks = aligner.apply_auto_align(template, img.copy())

            all_mean_values, all_strip_values_arrays, all_standard_deviation_values = (
                [],
                [],
                [],
            )
            for field_block in template.field_blocks:
                for bubble_group in field_block.traverse_bubbles:
                    strip_values = self.process_bubble_group(
                        img,
                        bubble_group,
                        field_block,
                        *field_block.bubble_dimensions,
                        all_mean_values,
                        all_standard_deviation_values,
                    )
                    all_strip_values_arrays.append(strip_values)

            global_std_thresh = threshhold_calculator.get_global_threshold(
                all_standard_deviation_values
            )
            global_thr = threshhold_calculator.get_global_threshold(
                all_mean_values, looseness=4
            )

            logger.info(
                f"Thresholding:\tglobal_thr: {round(global_thr, 2)} \tglobal_std_THR: {round(global_std_thresh, 2)}"
            )

            omr_response, final_marked, multi_marked_count = self.process_field_blocks(
                final_marked,
                template,
                all_mean_values,
                all_strip_values_arrays,
                global_thr,
                global_std_thresh,
                threshhold_calculator,
                all_standard_deviation_values,
            )
            cropped_name = self.crop_name(final_marked)

            return omr_response, final_marked, cropped_name, multi_marked_count

        except Exception as e:
            raise e
