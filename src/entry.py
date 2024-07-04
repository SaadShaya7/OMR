from pathlib import Path
import cv2
from pyzbar.pyzbar import decode

from utils.image import ImageUtils
from template import Template


class WrongSampleException(Exception):
    pass


def entry_point(image_path, template_path, sample_id):

    template = Template(Path(template_path))

    in_omr = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if in_omr is None:
        raise Exception(f"Could not read the provided image")

    in_omr = template.image_instance_ops.apply_preprocessors(image_path, in_omr)

    if in_omr is None:
        raise Exception(f"Failure after applying processors")

    if not sample_id == read_sample_id(in_omr, template):

        raise WrongSampleException("Wrong sample ")

    (omr_response, final_marked, cropped_name, multi_marked_count) = (
        template.image_instance_ops.read_omr_response(template, image=in_omr)
    )

    return final_marked, omr_response, cropped_name, multi_marked_count


def read_sample_id(original_image, template) -> str:

    original_image = ImageUtils.resize_util(
        original_image, template.page_dimensions[0], template.page_dimensions[1]
    )
    if original_image.max() > original_image.min():
        original_image = ImageUtils.normalize_util(original_image)

    origin = (0, 630)
    width = 130
    height = 130
    end_point = (origin[0] + width, origin[1] + height)
    cropped = original_image[origin[1] : end_point[1], origin[0] : end_point[0]]

    decoded = next(iter(decode(cropped)), None)

    data = None if decoded == None else decoded.data.decode("utf-8")

    print(data)
    # cv2.imshow("sdf", cropped)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return data
