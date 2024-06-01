from pathlib import Path
from flask import Flask, request, jsonify
import json
import os
from src.entry import entry_point
import tempfile
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)


@app.route("/processImage", methods=["POST"])
def process_image():
    if "image" not in request.files or "template" not in request.form:
        return jsonify({"error": "Image and template are required"}), 400

    image_file = request.files["image"]
    template_str = request.form["template"]

    try:
        template = json.loads(template_str)
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON template"}), 400

    temp_image_path = None
    temp_template_path = None

    try:
        # Save the image to a temporary file
        temp_image_file = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        temp_image_path = temp_image_file.name
        temp_image_file.write(image_file.read())
        temp_image_file.close()

        # Save the template to a temporary file
        temp_template_file = tempfile.NamedTemporaryFile(delete=False, suffix=".json")
        temp_template_path = temp_template_file.name
        with open(temp_template_path, "w") as f:
            json.dump(template, f)

        # Process the image using the template
        entry_point(temp_image_path, temp_template_path)

        return jsonify({"message": "Image processed successfully"}), 200

    except Exception as e:
        logging.error("An error occurred: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500

    # finally:
    #     # Clean up temporary files
    #     if temp_image_path and os.path.exists(temp_image_path):
    #         os.remove(temp_image_path)
    #     if temp_template_path and os.path.exists(temp_template_path):
    #         os.remove(temp_template_path)


if __name__ == "__main__":
    app.run(debug=True)
