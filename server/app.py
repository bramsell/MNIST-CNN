import os
import sys

# Allow running this file directly (e.g. VS Code Run button) in addition to
# `py -m server.app` — ensures the project root is always on sys.path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from server.inference import Predictor

CLIENT_DIR = os.path.join(os.path.dirname(__file__), "..", "client")

app = Flask(__name__, static_folder=None)
CORS(app)  # allow requests from any origin (browser clients on the LAN)

# Load model once at startup — shared across all requests
predictor = Predictor()


@app.route("/", methods=["GET"])
def index():
    """Serve the drawing client."""
    return send_from_directory(CLIENT_DIR, "index.html")


@app.route("/<path:filename>", methods=["GET"])
def static_client(filename):
    """Serve client static files (style.css, app.js)."""
    return send_from_directory(CLIENT_DIR, filename)


@app.route("/health", methods=["GET"])
def health():
    """Quick check that the server is up and the model is loaded."""
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts a raw image file (PNG recommended) as multipart/form-data.
    Returns JSON with the predicted digit and confidence scores.

    Example curl:
        curl -X POST http://<server-ip>:5000/predict \
             -F "image=@digit.png"
    """
    if "image" not in request.files:
        return jsonify({"error": "No image field in request"}), 400

    image_bytes = request.files["image"].read()
    if not image_bytes:
        return jsonify({"error": "Image file is empty"}), 400

    result = predictor.predict(image_bytes)
    return jsonify(result)
