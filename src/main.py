# main.py
from flask import Flask, request, jsonify
import logging
from process import process_image_direct

app = Flask(__name__)

logging.basicConfig(level=logging.DEBUG)

@app.route("/processImage", methods=["POST"])
def process_image():
    image_file = request.files.get("image")
    template_str = request.form.get("template")

    if not image_file or not template_str:
        return jsonify({"error": "Image and template are required"}), 400

    try:
        response = process_image_direct(image_file, template_str)
        return jsonify({"message": "Image processed successfully", "response": response}), 200
    except Exception as e:
        logging.error("An error occurred: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=55001)
