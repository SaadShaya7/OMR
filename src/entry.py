"""

 OMRChecker

 Author: Udayraj Deshmukh
 Github: https://github.com/Udayraj123

"""

from pathlib import Path
import cv2
from defaults import CONFIG_DEFAULTS
from utils.interaction import InteractionUtils
from template import Template


def entry_point(image_path, template_path):
    return process_image(image_path, template_path)


def process_image(image_path, template_path):
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

    (omr_response, final_marked, cropped_name) = (
        template.image_instance_ops.read_omr_response(template, image=in_omr)
    )

    return final_marked, omr_response, cropped_name


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
