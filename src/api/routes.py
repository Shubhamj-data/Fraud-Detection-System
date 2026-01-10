import os
import json
import pandas as pd
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
UNIQUE_VALUES_PATH = os.path.join(BASE_DIR, "artifacts", "unique_values.json")

try:
    with open(UNIQUE_VALUES_PATH) as f:
        UNIQUE_VALUES = json.load(f)
except FileNotFoundError:
    UNIQUE_VALUES = {}
    print("⚠️ unique_values.json not found — dropdowns disabled")

from src.preprocessing.preprocessor_batch import batch_transform

from flask import Blueprint, request, jsonify, current_app, render_template
import sqlite3

from src.api.sheets_logger import log_to_google_sheets


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
    return render_template(
        "index.html",
        options=UNIQUE_VALUES
    )


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

        log_to_google_sheets(input_data, prediction, probability, "v1.0")



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
        input_data = {
            "transaction type": request.form["transaction type"],
            "merchant_category": request.form["merchant_category"],
            "transaction_status": "SUCCESS",
            "sender_age_group": request.form["sender_age_group"],
            "receiver_age_group": request.form["receiver_age_group"],
            "sender_state": request.form["sender_state"],
            "sender_bank": request.form["sender_bank"],
            "receiver_bank": request.form["receiver_bank"],
            "device_type": request.form["device_type"],
            "network_type": request.form["network_type"],
            "hour_of_day": int(request.form["hour_of_day"]),
            "day_of_week": request.form["day_of_week"],
            "is_weekend": int(request.form["is_weekend"]),
            "amount (INR)": float(request.form["amount (INR)"])
        }

        df = pd.DataFrame([input_data])

        X = batch_transform(df)

        model = current_app.model

        prediction = int(model.predict(X)[0])
        fraud_prob = float(model.predict_proba(X)[0][1])

        # 🔥 ADD THIS LINE
        log_to_google_sheets(input_data, prediction, fraud_prob, "v1.0")
        
        prediction_label = "Fraud" if prediction == 1 else "Legitimate"

        return render_template(
            "index.html",
            result={
                "prediction": prediction_label,
                "probability": f"{fraud_prob:.4f}"
            },
            options=UNIQUE_VALUES
        )

    except Exception as e:
        return render_template(
            "index.html",
            result={
                "prediction": "Error",
                "probability": str(e)
            },
            options=UNIQUE_VALUES
        )


@api_blueprint.route("/history", methods=["GET"])
def history():
    conn = sqlite3.connect("predictions.db")
    cursor = conn.cursor()

    cursor.execute("""
        SELECT timestamp, transaction_json, fraud_prediction, fraud_probability
        FROM prediction_logs
        ORDER BY timestamp DESC
        LIMIT 50
    """)

    rows = cursor.fetchall()
    conn.close()

    history_data = []
    for row in rows:
        history_data.append({
            "timestamp": row[0],
            "transaction": row[1],
            "prediction": "Fraud" if row[2] == 1 else "Not Fraud",
            "probability": round(row[3], 4)
        })

    return render_template("history.html", history=history_data)
