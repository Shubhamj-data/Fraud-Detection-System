from flask import Blueprint, request, jsonify, current_app, render_template

# preprocessing
from src.preprocessing.schema import validate_input
from src.preprocessing.preprocessor_transform import transform_input

# database
from src.api.db import log_prediction

# define blueprint
api_blueprint = Blueprint("api", __name__)


# -----------------------
# Health check
# -----------------------
@api_blueprint.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


# -----------------------
# Home page (HTML form)
# -----------------------
@api_blueprint.route("/", methods=["GET"])
def home():
    return render_template("index.html")


# -----------------------
# JSON API prediction
# -----------------------
@api_blueprint.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()

        if input_data is None:
            return jsonify({"error": "Invalid or missing JSON"}), 400

        validate_input(input_data)

        X = transform_input(input_data)
        model = current_app.model

        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])

        log_prediction(input_data, prediction, probability, "v1.0")

        return jsonify({
            "fraud_prediction": prediction,
            "fraud_probability": round(probability, 4),
            "model_version": "v1.0"
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------
# HTML form submission
# -----------------------
@api_blueprint.route("/predict-form", methods=["POST"])
def predict_form():
    try:
        input_data = dict(request.form)

        # type conversion
        input_data["amount (INR)"] = float(input_data["amount (INR)"])
        input_data["hour_of_day"] = int(input_data["hour_of_day"])
        input_data["is_weekend"] = int(input_data["is_weekend"])

        validate_input(input_data)

        X = transform_input(input_data)
        model = current_app.model

        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])

        log_prediction(input_data, prediction, probability, "v1.0")

        result = {
            "prediction": "Fraud" if prediction == 1 else "Not Fraud",
            "probability": round(probability, 4)
        }

        return render_template("index.html", result=result)

    except Exception as e:
        return render_template(
            "index.html",
            result={"prediction": "Error", "probability": str(e)}
        )
