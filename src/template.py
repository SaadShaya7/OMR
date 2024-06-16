import os
from constants import FIELD_TYPES
from core import ImageInstanceOps
from logger import logger
from processors.manager import PROCESSOR_MANAGER
from utils.parsing import (
    open_template_with_defaults,
    parse_fields,
)


class Template:
    def __init__(self, template_path, tuning_config):
        self.path = template_path
        self.image_instance_ops = ImageInstanceOps(tuning_config)

        json_object = open_template_with_defaults(template_path)

        self.bubble_dimensions = [14, 14]
        self.page_dimensions = [524, 772]
        self.global_empty_val = None

        (
            field_blocks_object,
            self.options,
        ) = map(
            json_object.get,
            [
                "fieldBlocks",
                "options",
            ],
        )

        self.setup_pre_processors()
        self.setup_field_blocks(field_blocks_object)

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
            correct_answers,
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
                "correctAnswers",
            ],
        )
        self.parsed_field_labels = parse_fields(
            f"Field Block Labels: {self.name}", field_labels
        )

        self.origin = origin
        self.bubble_dimensions = bubble_dimensions
        self.correct_answers = correct_answers
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
