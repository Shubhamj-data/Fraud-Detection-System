from flask import Flask, render_template, request, current_app
import joblib
import os

# -------------------------
# Project Imports
# -------------------------
from src.preprocessing.preprocessor import preprocess_input

from src.api.google_sheets_logger import log_transaction_to_sheet

# -------------------------
# Flask App Initialization
# -------------------------
app = Flask(__name__)

# -------------------------
# Load Model ONCE at startup
# -------------------------
MODEL_PATHS = [
    "artifacts/best_model.pkl",
    "artifacts/model.pkl"
]

model_path = None
for p in MODEL_PATHS:
    if os.path.exists(p):
        model_path = p
        break

if not model_path:
    raise FileNotFoundError("❌ No trained model found. Run main.py first.")

app.model = joblib.load(model_path)
print(f"✅ Model loaded successfully from {model_path}")

# -------------------------
# Load dropdown values SAFELY
# -------------------------
try:
    from src.preprocessing.preprocessor import UNIQUE_VALUES
except Exception:
    UNIQUE_VALUES = {}
    print("⚠️ UNIQUE_VALUES not found — dropdowns disabled")

# -------------------------
# Home Page
# -------------------------
@app.route("/", methods=["GET"])
def home():
    return render_template(
        "index.html",
        options=UNIQUE_VALUES
    )

# -------------------------
# Prediction via HTML Form
# -------------------------
@app.route("/predict-form", methods=["POST"])
def predict_form():
    try:
        # -------------------------
        # Read form input
        # -------------------------
        input_data = {
            "transaction type": request.form.get("transaction type"),
            "merchant_category": request.form.get("merchant_category"),
            "amount (INR)": float(request.form.get("amount (INR)", 0)),

            "transaction_status": "SUCCESS",

            "sender_age_group": request.form.get("sender_age_group"),
            "receiver_age_group": request.form.get("receiver_age_group"),

            "sender_state": request.form.get("sender_state"),
            "sender_bank": request.form.get("sender_bank"),
            "receiver_bank": request.form.get("receiver_bank"),

            "device_type": request.form.get("device_type"),
            "network_type": request.form.get("network_type"),

            "hour_of_day": int(request.form.get("hour_of_day")),
            "day_of_week": request.form.get("day_of_week"),
            "is_weekend": int(request.form.get("is_weekend")),
        }

        # -------------------------
        # Preprocess input
        # -------------------------
        X = preprocess_input(input_data)

        # -------------------------
        # Predict
        # -------------------------
        model = current_app.model
        prediction = int(model.predict(X)[0])
        probability = float(model.predict_proba(X)[0][1])

        # -------------------------
        # Log to Google Sheets
        # -------------------------
        log_transaction_to_sheet(
            input_data=input_data,
            prediction=prediction,
            probability=probability,
            model_version="v1.0"
        )

        # -------------------------
        # Render result
        # -------------------------
        return render_template(
            "index.html",
            result={
                "prediction": "Fraud" if prediction == 1 else "Legitimate",
                "probability": f"{probability:.6f}"
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

# -------------------------
# App Runner
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)
