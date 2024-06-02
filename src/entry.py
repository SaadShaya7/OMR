from pathlib import Path
import cv2
from src.defaults import CONFIG_DEFAULTS
from src.logger import logger
from src.template import Template
from src.utils.file import Paths, setup_dirs_for_paths, setup_outputs_for_template
from src.utils.interaction import Stats
from src.utils.parsing import get_concatenated_response


STATS = Stats()


def entry_point(image_path, template_path):
    return process_image(image_path, template_path)


def process_image(image_path, template_path):
    tuning_config = CONFIG_DEFAULTS

    # Load template
    template = Template(Path(template_path), tuning_config)

    # Process the image
    output_dir = Path("outputs")
    paths = Paths(output_dir)
    setup_dirs_for_paths(paths)
    outputs_namespace = setup_outputs_for_template(paths, template)

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
    save_dir = outputs_namespace.paths.save_marked_dir
    (
        response_dict,
        final_marked,
        multi_marked,
        _,
    ) = template.image_instance_ops.read_omr_response(
        template, image=in_omr, name=file_id, save_dir=save_dir
    )

    omr_response = get_concatenated_response(response_dict, template)

    score = 0

    results_line = [file_id, str(image_path), str(save_dir), score] + list(
        omr_response.values()
    )
    # pd.DataFrame([results_line], dtype=str).to_csv(
    #     outputs_namespace.files_obj["Results"],
    #     mode="a",
    #     # quoting=QUOTE_NONNUMERIC,
    #     header=False,
    #     index=False,
    # )

    logger.info(f"Processed image {file_id} with score {score}")

    return {"file_id": file_id, "score": score, "omr_response": omr_response}
