import pandas as pd
import joblib

from src.preprocessing.schema import REQUIRED_FEATURES

ARTIFACT_DIR = "artifacts"


def batch_transform(data: pd.DataFrame) -> pd.DataFrame:
    """
    Fast batch transformation for model training.
    Uses frozen preprocessing artifacts.
    """

    # 1. Keep only required features
    X = data[REQUIRED_FEATURES].copy()

    # 2. One-hot encode (no fitting, just transform)
    X_encoded = pd.get_dummies(X, drop_first=True)

    # 3. Load feature columns from training
    feature_columns = joblib.load(f"{ARTIFACT_DIR}/feature_columns.pkl")

    # 4. Align columns
    X_encoded = X_encoded.reindex(
        columns=feature_columns,
        fill_value=0
    )

    # 5. Load scaler and scale amount
    scaler = joblib.load(f"{ARTIFACT_DIR}/scaler.pkl")

    X_encoded["amount_scaled"] = scaler.transform(
        X_encoded[["amount (INR)"]]
    )

    X_encoded.drop(columns=["amount (INR)"], inplace=True)

    return X_encoded
