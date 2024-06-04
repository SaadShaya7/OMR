import os
from constants import FIELD_TYPES
from core import ImageInstanceOps
import cv2
from logger import logger
import numpy as np
from utils.parsing import (
    custom_sort_output_columns,
    open_template_with_defaults,
    parse_fields,
)

from src.processors.manager import PROCESSOR_MANAGER
from src.utils.image import ImageUtils


class Template:
    def __init__(self, template_path, tuning_config):
        self.path = template_path
        self.image_instance_ops = ImageInstanceOps(tuning_config)

        json_object = open_template_with_defaults(template_path)

        self.bubble_dimensions = [14, 14]
        self.page_dimensions = [524, 772]
        self.global_empty_val = ""

        (
            custom_labels_object,
            field_blocks_object,
            output_columns_array,
            self.options,
        ) = map(
            json_object.get,
            [
                "customLabels",
                "fieldBlocks",
                "outputColumns",
                "options",
            ],
        )

        self.parse_output_columns(output_columns_array)
        self.setup_pre_processors()
        self.setup_field_blocks(field_blocks_object)
        self.parse_custom_labels(custom_labels_object)

        non_custom_columns, all_custom_columns = (
            list(self.non_custom_labels),
            list(custom_labels_object.keys()),  #
        )

        if len(self.output_columns) == 0:
            self.fill_output_columns(non_custom_columns, all_custom_columns)

        self.validate_template_columns(non_custom_columns, all_custom_columns)

    def parse_output_columns(self, output_columns_array):
        self.output_columns = parse_fields(f"Output Columns", output_columns_array)

    def setup_pre_processors(self):
        # Define the path to the marker image in the constants directory
        marker_image_path = os.path.join(
            os.path.dirname(__file__), "constants", "marker_image.jpg"
        )

        # Directly create an instance of CropOnMarkers with the hardcoded image path
        ProcessorClass = PROCESSOR_MANAGER.processors["CropOnMarkers"]
        pre_processor_instance = ProcessorClass(
            options={
                "relativePath": marker_image_path,
                "sheetToMarkerWidthRatio": 21,
            },
            relative_dir=os.path.dirname(__file__),
            image_instance_ops=self.image_instance_ops,
        )
        self.pre_processors = [pre_processor_instance]

    def setup_field_blocks(self, field_blocks_object):
        # Add field_blocks
        self.field_blocks = []
        self.all_parsed_labels = set()
        for block_name, field_block_object in field_blocks_object.items():
            self.parse_and_add_field_block(block_name, field_block_object)

    def parse_custom_labels(self, custom_labels_object):
        all_parsed_custom_labels = set()
        self.custom_labels = {}
        for custom_label, label_strings in custom_labels_object.items():
            parsed_labels = parse_fields(f"Custom Label: {custom_label}", label_strings)
            parsed_labels_set = set(parsed_labels)
            self.custom_labels[custom_label] = parsed_labels

            missing_custom_labels = sorted(
                parsed_labels_set.difference(self.all_parsed_labels)
            )
            if len(missing_custom_labels) > 0:
                logger.critical(
                    f"For '{custom_label}', Missing labels - {missing_custom_labels}"
                )
                raise Exception(
                    f"Missing field block label(s) in the given template for {missing_custom_labels} from '{custom_label}'"
                )

            if not all_parsed_custom_labels.isdisjoint(parsed_labels_set):
                # Note: this can be made a warning, but it's a choice
                logger.critical(
                    f"field strings overlap for labels: {label_strings} and existing custom labels: {all_parsed_custom_labels}"
                )
                raise Exception(
                    f"The field strings for custom label '{custom_label}' overlap with other existing custom labels"
                )

            all_parsed_custom_labels.update(parsed_labels)

        self.non_custom_labels = self.all_parsed_labels.difference(
            all_parsed_custom_labels
        )

    def fill_output_columns(self, non_custom_columns, all_custom_columns):
        all_template_columns = non_custom_columns + all_custom_columns
        # Typical case: sort alpha-numerical (natural sort)
        self.output_columns = sorted(
            all_template_columns, key=custom_sort_output_columns
        )

    def validate_template_columns(self, non_custom_columns, all_custom_columns):
        output_columns_set = set(self.output_columns)
        all_custom_columns_set = set(all_custom_columns)

        missing_output_columns = sorted(
            output_columns_set.difference(all_custom_columns_set).difference(
                self.all_parsed_labels
            )
        )
        if len(missing_output_columns) > 0:
            logger.critical(f"Missing output columns: {missing_output_columns}")
            raise Exception(
                f"Some columns are missing in the field blocks for the given output columns"
            )

        all_template_columns_set = set(non_custom_columns + all_custom_columns)
        missing_label_columns = sorted(
            all_template_columns_set.difference(output_columns_set)
        )
        if len(missing_label_columns) > 0:
            logger.warning(
                f"Some label columns are not covered in the given output columns: {missing_label_columns}"
            )

    def parse_and_add_field_block(self, block_name, field_block_object):
        field_block_object = self.pre_fill_field_block(field_block_object)
        block_instance = FieldBlock(block_name, field_block_object)
        self.field_blocks.append(block_instance)
        self.validate_parsed_labels(field_block_object["fieldLabels"], block_instance)

    def pre_fill_field_block(self, field_block_object):
        if "fieldType" in field_block_object:
            field_block_object = {
                **field_block_object,
                **FIELD_TYPES[field_block_object["fieldType"]],
            }
        else:
            field_block_object = {**field_block_object, "fieldType": "__CUSTOM__"}

        return {
            "direction": "vertical",
            "emptyValue": self.global_empty_val,
            "bubbleDimensions": self.bubble_dimensions,
            **field_block_object,
        }

    def validate_parsed_labels(self, field_labels, block_instance):
        parsed_field_labels, block_name = (
            block_instance.parsed_field_labels,
            block_instance.name,
        )
        field_labels_set = set(parsed_field_labels)
        if not self.all_parsed_labels.isdisjoint(field_labels_set):
            # Note: in case of two fields pointing to same column, use a custom column instead of same field labels.
            logger.critical(
                f"An overlap found between field string: {field_labels} in block '{block_name}' and existing labels: {self.all_parsed_labels}"
            )
            raise Exception(
                f"The field strings for field block {block_name} overlap with other existing fields"
            )
        self.all_parsed_labels.update(field_labels_set)

        page_width, page_height = self.page_dimensions
        block_width, block_height = block_instance.dimensions
        [block_start_x, block_start_y] = block_instance.origin

        block_end_x, block_end_y = (
            block_start_x + block_width,
            block_start_y + block_height,
        )

        if (
            block_end_x >= page_width
            or block_end_y >= page_height
            or block_start_x < 0
            or block_start_y < 0
        ):
            raise Exception(
                f"Overflowing field block '{block_name}' with origin {block_instance.origin} and dimensions {block_instance.dimensions} in template with dimensions {self.page_dimensions}"
            )

    def __str__(self):
        return str(self.path)


class FieldBlock:
    def __init__(self, block_name, field_block_object):
        self.name = block_name
        self.shift = 0
        self.setup_field_block(field_block_object)

    def setup_field_block(self, field_block_object):
        # case mapping
        (
            bubble_dimensions,
            bubble_values,
            bubbles_gap,
            direction,
            field_labels,
            field_type,
            labels_gap,
            origin,
            self.empty_val,
        ) = map(
            field_block_object.get,
            [
                "bubbleDimensions",
                "bubbleValues",
                "bubblesGap",
                "direction",
                "fieldLabels",
                "fieldType",
                "labelsGap",
                "origin",
                "emptyValue",
            ],
        )
        self.parsed_field_labels = parse_fields(
            f"Field Block Labels: {self.name}", field_labels
        )
        self.origin = origin
        self.bubble_dimensions = bubble_dimensions
        self.calculate_block_dimensions(
            bubble_dimensions,
            bubble_values,
            bubbles_gap,
            direction,
            labels_gap,
        )
        self.generate_bubble_grid(
            bubble_values,
            bubbles_gap,
            direction,
            field_type,
            labels_gap,
        )

    def calculate_block_dimensions(
        self,
        bubble_dimensions,
        bubble_values,
        bubbles_gap,
        direction,
        labels_gap,
    ):
        _h, _v = (1, 0) if (direction == "vertical") else (0, 1)

        values_dimension = int(
            bubbles_gap * (len(bubble_values) - 1) + bubble_dimensions[_h]
        )
        fields_dimension = int(
            labels_gap * (len(self.parsed_field_labels) - 1) + bubble_dimensions[_v]
        )
        self.dimensions = (
            [fields_dimension, values_dimension]
            if (direction == "vertical")
            else [values_dimension, fields_dimension]
        )

    def generate_bubble_grid(
        self,
        bubble_values,
        bubbles_gap,
        direction,
        field_type,
        labels_gap,
    ):
        _h, _v = (1, 0) if (direction == "vertical") else (0, 1)
        self.traverse_bubbles = []
        # Generate the bubble grid
        lead_point = [float(self.origin[0]), float(self.origin[1])]
        for field_label in self.parsed_field_labels:
            bubble_point = lead_point.copy()
            field_bubbles = []
            for bubble_value in bubble_values:
                field_bubbles.append(
                    Bubble(bubble_point.copy(), field_label, field_type, bubble_value)
                )
                bubble_point[_h] += bubbles_gap
            self.traverse_bubbles.append(field_bubbles)
            lead_point[_v] += labels_gap


class Bubble:
    """
    Container for a Point Box on the OMR

    field_label is the point's property- field to which this point belongs to
    It can be used as a roll number column as well. (eg roll1)
    It can also correspond to a single digit of integer type Q (eg q5d1)
    """

    def __init__(self, pt, field_label, field_type, field_value):
        self.x = round(pt[0])
        self.y = round(pt[1])
        self.field_label = field_label
        self.field_type = field_type
        self.field_value = field_value

    def __str__(self):
        return str([self.x, self.y])


class CropOnMarkers:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        config = self.tuning_config
        marker_ops = self.options
        self.threshold_circles = []
        self.marker_path = os.path.join(
            self.relative_dir, marker_ops.get("relativePath", "omr_marker.jpg")
        )
        self.min_matching_threshold = marker_ops.get("min_matching_threshold", 0.3)
        self.max_matching_variation = marker_ops.get("max_matching_variation", 0.41)
        self.marker_rescale_range = tuple(
            int(r) for r in marker_ops.get("marker_rescale_range", (35, 100))
        )
        self.marker_rescale_steps = int(marker_ops.get("marker_rescale_steps", 10))
        self.apply_erode_subtract = marker_ops.get("apply_erode_subtract", True)
        self.marker = self.load_marker(marker_ops, config)

    def __str__(self):
        return self.marker_path

    def exclude_files(self):
        return [self.marker_path]

    def apply_filter(self, image, file_path):
        config = self.tuning_config
        image_instance_ops = self.image_instance_ops
        image_eroded_sub = ImageUtils.normalize_util(
            image
            if self.apply_erode_subtract
            else (image - cv2.erode(image, kernel=np.ones((5, 5)), iterations=5))
        )
        quads = {}
        h1, w1 = image_eroded_sub.shape[:2]
        midh, midw = h1 // 3, w1 // 2
        origins = [[0, 0], [midw, 0], [0, midh], [midw, midh]]
        quads[0] = image_eroded_sub[0:midh, 0:midw]
        quads[1] = image_eroded_sub[0:midh, midw:w1]
        quads[2] = image_eroded_sub[midh:h1, 0:midw]
        quads[3] = image_eroded_sub[midh:h1, midw:w1]
        image_eroded_sub[:, midw : midw + 2] = 255
        image_eroded_sub[midh : midh + 2, :] = 255
        best_scale, all_max_t = self.getBestMatch(image_eroded_sub)
        if best_scale is None:
            return None
        optimal_marker = ImageUtils.resize_util_h(
            self.marker, u_height=int(self.marker.shape[0] * best_scale)
        )
        _h, w = optimal_marker.shape[:2]
        centres = []
        sum_t, max_t = 0, 0
        quarter_match_log = "Matching Marker:  "
        for k in range(0, 4):
            res = cv2.matchTemplate(quads[k], optimal_marker, cv2.TM_CCOEFF_NORMED)
            max_t = res.max()
            quarter_match_log += f"Quarter{str(k + 1)}: {str(round(max_t, 3))}\t"
            if (
                max_t < self.min_matching_threshold
                or abs(all_max_t - max_t) >= self.max_matching_variation
            ):
                logger.error(
                    file_path,
                    "\nError: No circle found in Quad",
                    k + 1,
                    "\n\t min_matching_threshold",
                    self.min_matching_threshold,
                    "\t max_matching_variation",
                    self.max_matching_variation,
                    "\t max_t",
                    max_t,
                    "\t all_max_t",
                    all_max_t,
                )
                return None
            pt = np.argwhere(res == max_t)[0]
            pt = [pt[1], pt[0]]
            pt[0] += origins[k][0]
            pt[1] += origins[k][1]
            image = cv2.rectangle(
                image, tuple(pt), (pt[0] + w, pt[1] + _h), (150, 150, 150), 2
            )
            image_eroded_sub = cv2.rectangle(
                image_eroded_sub,
                tuple(pt),
                (pt[0] + w, pt[1] + _h),
                (50, 50, 50) if self.apply_erode_subtract else (155, 155, 155),
                4,
            )
            centres.append([pt[0] + w / 2, pt[1] + _h / 2])
            sum_t += max_t
        logger.info(quarter_match_log)
        logger.info(f"Optimal Scale: {best_scale}")
        self.threshold_circles.append(sum_t / 4)
        image = ImageUtils.four_point_transform(image, np.array(centres))
        # image_instance_ops.append_save_img(2, image_eroded_sub)
        if config.outputs.show_image_level >= 2 and config.outputs.show_image_level < 4:
            image_eroded_sub = ImageUtils.resize_util_h(
                image_eroded_sub, image.shape[0]
            )
            image_eroded_sub[:, -5:] = 0
            h_stack = np.hstack((image_eroded_sub, image))

        return image

    def load_marker(self, marker_ops, config):
        if not os.path.exists(self.marker_path):
            logger.error(
                "Marker not found at path provided in template:",
                self.marker_path,
            )
            exit(31)
        marker = cv2.imread(self.marker_path, cv2.IMREAD_GRAYSCALE)
        if "sheetToMarkerWidthRatio" in marker_ops:
            marker = ImageUtils.resize_util(
                marker,
                config.dimensions.processing_width
                / int(marker_ops["sheetToMarkerWidthRatio"]),
            )
        marker = cv2.GaussianBlur(marker, (5, 5), 0)
        marker = cv2.normalize(
            marker, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX
        )
        if self.apply_erode_subtract:
            marker -= cv2.erode(marker, kernel=np.ones((5, 5)), iterations=5)
        return marker

    def getBestMatch(self, image_eroded_sub):
        config = self.tuning_config
        descent_per_step = (
            self.marker_rescale_range[1] - self.marker_rescale_range[0]
        ) // self.marker_rescale_steps
        _h, _w = self.marker.shape[:2]
        res, best_scale = None, None
        all_max_t = 0
        for r0 in np.arange(
            self.marker_rescale_range[1],
            self.marker_rescale_range[0],
            -1 * descent_per_step,
        ):
            s = float(r0 * 1 / 100)
            if s == 0.0:
                continue
            rescaled_marker = ImageUtils.resize_util_h(
                self.marker, u_height=int(_h * s)
            )
            res = cv2.matchTemplate(
                image_eroded_sub, rescaled_marker, cv2.TM_CCOEFF_NORMED
            )
            max_t = res.max()
            if all_max_t < max_t:
                best_scale, all_max_t = s, max_t
        if all_max_t < self.min_matching_threshold:
            logger.warning(
                "\tTemplate matching too low! Consider rechecking preProcessors applied before this."
            )
        if best_scale is None:
            logger.warning(
                "No matchings for given scaleRange:", self.marker_rescale_range
            )
        return best_scale, all_max_t
