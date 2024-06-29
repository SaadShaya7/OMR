from pathlib import Path
import cv2
from defaults import CONFIG_DEFAULTS
from utils.image import ImageUtils
from utils.interaction import InteractionUtils
from template import Template
from pyzbar.pyzbar import decode


class WrongSampleException(Exception):
    pass


def entry_point(image_path, template_path, sample_id):
    tuning_config = CONFIG_DEFAULTS

    # Load template
    template = Template(Path(template_path), tuning_config)

    in_omr = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    # show_template_layouts(str(image_path), template, tuning_config)

    if in_omr is None:
        raise Exception(f"Could not read the provided image")

    in_omr = template.image_instance_ops.apply_preprocessors(
        image_path, in_omr, template
    )

    if in_omr is None:
        raise Exception(f"Failure after applying processors")

    if not sample_id == read_sample_id(in_omr, template):
        raise WrongSampleException("Wrong sample ")

    (omr_response, final_marked, cropped_name, multi_marked_count) = (
        template.image_instance_ops.read_omr_response(template, image=in_omr)
    )

    return final_marked, omr_response, cropped_name, multi_marked_count


def show_template_layouts(file_path, template, tuning_config):
    in_omr = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    in_omr = cv2.flip(in_omr, 1)
    in_omr = template.image_instance_ops.apply_preprocessors(
        file_path, in_omr, template
    )
    template_layout = template.image_instance_ops.draw_template_layout(
        in_omr, template, shifted=False, border=2
    )
    InteractionUtils.show(
        f"Template Layout", template_layout, 1, 1, config=tuning_config
    )


def read_sample_id(original_image, template) -> str:

    original_image = ImageUtils.resize_util(
        original_image, template.page_dimensions[0], template.page_dimensions[1]
    )
    if original_image.max() > original_image.min():
        original_image = ImageUtils.normalize_util(original_image)

    origin = (20, 675)
    width = 80
    height = 80
    end_point = (origin[0] + width, origin[1] + height)
    cropped = original_image[origin[1] : end_point[1], origin[0] : end_point[0]]

    decoded = next(iter(decode(cropped)), None)
    if decoded == None:
        return None

    return decoded.data.decode("utf-8")
