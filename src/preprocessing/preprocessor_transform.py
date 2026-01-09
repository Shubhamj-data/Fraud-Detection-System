import pandas as pd
import joblib

from src.preprocessing.schema import REQUIRED_FEATURES, validate_input

ARTIFACT_DIR = "artifacts"


def transform_input(input_data: dict) -> pd.DataFrame:
    """
    Transform a single transaction input into model-ready format.
    """

    # 1. Validate input
    validate_input(input_data)

    # 2. Convert dict to DataFrame
    df = pd.DataFrame([input_data], columns=REQUIRED_FEATURES)

    # 3. One-hot encode
    df_encoded = pd.get_dummies(df, drop_first=True)

    # 4. Load training feature columns
    feature_columns = joblib.load(f"{ARTIFACT_DIR}/feature_columns.pkl")

    # 5. Align columns (add missing, reorder)
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    # 6. Load scaler and scale amount
    scaler = joblib.load(f"{ARTIFACT_DIR}/scaler.pkl")
    df_encoded["amount_scaled"] = scaler.transform(
        df_encoded[["amount (INR)"]]
    )

    df_encoded.drop(columns=["amount (INR)"], inplace=True)

    return df_encoded
