import base64
import io
import json
import os
import tempfile
import logging
from flask import Flask, request, jsonify
from PIL import Image
from entry import entry_point

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)


@app.route("/processImage", methods=["POST"])
def process_image():
    logging.info("Received a request to process an image")
    image_file = request.files.get("image")
    template_str = request.form.get("template")

    if not image_file or not template_str:
        logging.error("Image and template are required")
        return jsonify({"error": "Image and template are required"}), 400

    try:
        logging.info("Parsing the template JSON")
        template = json.loads(template_str)
    except json.JSONDecodeError:
        logging.error("Invalid JSON template")
        return jsonify({"error": "Invalid JSON template"}), 400

    temp_image_path = None
    temp_template_path = None

    try:
        logging.info("Creating temporary files for the image and template")
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".jpg"
        ) as temp_image_file:
            temp_image_file.write(image_file.read())
            temp_image_path = temp_image_file.name

        with tempfile.NamedTemporaryFile(
            delete=False, suffix=".json", mode="w"
        ) as temp_template_file:
            json.dump(template, temp_template_file)
            temp_template_path = temp_template_file.name

        # Process the image using the template
        logging.info("Processing the image using the template")
        ndArrayResponse = entry_point(temp_image_path, temp_template_path)
        processed_image = Image.fromarray(ndArrayResponse)
        buffer = io.BytesIO()
        processed_image.save(buffer, format="JPEG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")

        # Return the base64 encoded image as part of the JSON response
        logging.info("Image processed successfully")
        return (
            jsonify(
                {
                    "message": "Image processed successfully",
                    "response": image_base64,
                }
            ),
            200,
        )

    except Exception as e:
        logging.error("An error occurred: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean up temporary files
        logging.info("Cleaning up temporary files")
        if temp_image_path:
            os.remove(temp_image_path)
        if temp_template_path:
            os.remove(temp_template_path)


if __name__ == "__main__":
    app.run(debug=True, port=55001)
