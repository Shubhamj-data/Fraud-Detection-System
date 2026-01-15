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
    try:
        from src.preprocessing.preprocessor import UNIQUE_VALUES
    except:
        UNIQUE_VALUES = {}

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
        input_data = {
            "transaction type": request.form["transaction type"],
            "merchant_category": request.form["merchant_category"],
            "amount (INR)": float(request.form["amount (INR)"]),
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
        }

        X = preprocess_input(input_data)

        model = current_app.model
        pred = int(model.predict(X)[0])
        prob = float(model.predict_proba(X)[0][1])

        log_transaction_to_sheet(
            input_data,
            pred,
            prob,
            model_version="v1.0"
        )

        return render_template(
            "index.html",
            result={
                "prediction": "Fraud" if pred == 1 else "Legitimate",
                "probability": f"{prob:.6f}"
            },
            options={}
        )

    except Exception as e:
        return render_template(
            "index.html",
            result={
                "prediction": "Error",
                "probability": str(e)
            },
            options={}
        )

# -------------------------
# App Runner
# -------------------------
if __name__ == "__main__":
    app.run(debug=True)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
