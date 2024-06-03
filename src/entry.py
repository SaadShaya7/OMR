from pathlib import Path
import cv2
from defaults import CONFIG_DEFAULTS
from logger import logger
from template import Template
from utils.interaction import Stats
from utils.parsing import get_concatenated_response


STATS = Stats()


def entry_point(image_path, template_path):
    return process_image(image_path, template_path)


def process_image(image_path, template_path):
    tuning_config = CONFIG_DEFAULTS

    # Load template
    template = Template(Path(template_path), tuning_config)

    in_omr = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if in_omr is None:
        raise Exception(f"Could not read the provided image")

    template.image_instance_ops.reset_all_save_img()
    template.image_instance_ops.append_save_img(1, in_omr)
    in_omr = template.image_instance_ops.apply_preprocessors(
        image_path, in_omr, template
    )

    if in_omr is None:
        raise Exception(f"Failure after applying processors")

    file_id = Path(image_path).name
    # save_dir = outputs_namespace.paths.save_marked_dir
    (
        response_dict,
        final_marked,
        multi_marked,
        _,
    ) = template.image_instance_ops.read_omr_response(
        template, image=in_omr, name=file_id
    )

    omr_response = get_concatenated_response(response_dict, template)

    score = 0

    logger.info(f"Processed image {file_id} with score {score}")

    return final_marked
