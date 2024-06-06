import sys
import base64
import io
import json
from PIL import Image
from entry import entry_point


def process_image_direct(image_path, template_path):
    with open(image_path, "rb") as image_file:
        image = image_file.read()

    with open(template_path, "r") as template_file:
        template = json.load(template_file)

    ndArrayResponse, recognizedMarks = entry_point(image_path, template_path)
    processed_image = Image.fromarray(ndArrayResponse)
    buffer = io.BytesIO()
    processed_image.save(buffer, format="JPEG")
    image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

    return image_base64, recognizedMarks


if __name__ == "__main__":
    image_path = sys.argv[1]
    template_path = sys.argv[2]
    result, recognizedMarks = process_image_direct(image_path, template_path)
    print(result)
    print(recognizedMarks)
