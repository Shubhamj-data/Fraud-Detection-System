import pandas as pd
import joblib
import json
import os
import inspect

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

# -------------------------------------------------
# PATHS & CONSTANTS
# -------------------------------------------------
DATA_PATH = "data/processed/training_data.csv"
TARGET = "fraud_flag"
ARTIFACT_DIR = "artifacts"

os.makedirs(ARTIFACT_DIR, exist_ok=True)

# -------------------------------------------------
# ALLOWED COLUMNS (CRITICAL – prevents memory blowup)
# -------------------------------------------------
CATEGORICAL_COLS = [
    "transaction type",
    "merchant_category",
    "sender_state",
    "sender_bank",
    "receiver_bank",
    "device_type",
    "network_type",
    "day_of_week",
    "sender_age_group",
    "receiver_age_group",
]

NUM_COLS = ["amount (INR)", "hour_of_day", "is_weekend"]

ALL_FEATURE_COLS = NUM_COLS + CATEGORICAL_COLS

# -------------------------------------------------
# UNIQUE VALUES (for Flask dropdowns)
# -------------------------------------------------
UNIQUE_VALUES = {}

unique_path = f"{ARTIFACT_DIR}/unique_values.json"
if os.path.exists(unique_path):
    with open(unique_path) as f:
        UNIQUE_VALUES = json.load(f)
else:
    print("⚠️ UNIQUE_VALUES not found — dropdowns disabled")


# =================================================
# TRAINING PREPROCESSOR
# =================================================
def preprocess_data():
    print("⚙️ Preprocessing training data...")

    df = pd.read_csv(DATA_PATH, low_memory=False)

    if TARGET not in df.columns:
        raise ValueError("❌ fraud_flag column not found in training data")

    # keep only valid labels
    df = df[df[TARGET].isin([0, 1])]
    print(f"✅ Using {len(df)} labeled rows for training")

    # split X / y
    y = df[TARGET].astype(int)
    X = df[ALL_FEATURE_COLS].copy()

    # -------------------------------------------------
    # SAVE UNIQUE VALUES (for Flask UI)
    # -------------------------------------------------
    unique_values = {}
    for col in CATEGORICAL_COLS:
        unique_values[col] = sorted(X[col].dropna().unique().tolist())

    with open(unique_path, "w") as f:
        json.dump(unique_values, f)

    print("✅ UNIQUE_VALUES saved")

    # -------------------------------------------------
    # sklearn VERSION-SAFE OneHotEncoder
    # -------------------------------------------------
    ohe_params = inspect.signature(OneHotEncoder).parameters

    if "sparse_output" in ohe_params:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    else:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=True)

    # -------------------------------------------------
    # Column Transformer (MEMORY SAFE)
    # -------------------------------------------------
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUM_COLS),
            ("cat", ohe, CATEGORICAL_COLS),
        ],
        remainder="drop"
    )

    X_processed = preprocessor.fit_transform(X)

    # save preprocessor
    joblib.dump(preprocessor, f"{ARTIFACT_DIR}/preprocessor.pkl")

    print(f"📦 Final feature count: {X_processed.shape[1]}")
    print("✅ Preprocessing completed")

    return X_processed, y


# =================================================
# FLASK INPUT PREPROCESSING
# =================================================
def preprocess_input(input_data: dict):
    """
    Used by Flask for single prediction
    """
    preprocessor = joblib.load(f"{ARTIFACT_DIR}/preprocessor.pkl")

    df = pd.DataFrame([input_data])

    # enforce SAME column order as training
    df = df[ALL_FEATURE_COLS]

    return preprocessor.transform(df)
