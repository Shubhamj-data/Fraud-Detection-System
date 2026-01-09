# src/api/routes.py

from flask import Blueprint, request, jsonify, current_app

from src.preprocessing.schema import validate_input
from src.preprocessing.preprocessor_transform import transform_input
from src.api.db import log_prediction

api_blueprint = Blueprint("api", __name__)


@api_blueprint.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@api_blueprint.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Get JSON input
        input_data = request.get_json()

        if input_data is None:
            return jsonify({"error": "Invalid or missing JSON input"}), 400

        # 2. Validate input
        validate_input(input_data)

        # 3. Preprocess input (inference-time)
        X = transform_input(input_data)

        # 4. Load model from app context
        model = current_app.model

        # 5. Predict
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])

        # 6. Return response
        # log prediction
        log_prediction(input_data, prediction, probability, "v1.0")

        return jsonify({
            "fraud_prediction": prediction,
            "fraud_probability": round(probability, 4),
            "model_version": "v1.0"
            }), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400

    except Exception as e:
        return jsonify({"error": "Internal server error"}), 500
