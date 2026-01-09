import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from src.preprocessing.schema import REQUIRED_FEATURES


ARTIFACT_DIR = "artifacts"


def fit_preprocessor(csv_path: str):
    """
    Fit preprocessing objects on training data and save artifacts.
    """

    # Load data
    data = pd.read_csv(csv_path)

    # Keep only required features + target
    features = REQUIRED_FEATURES + ["fraud_flag"]
    data = data[features]

    # Drop target for preprocessing
    X = data.drop("fraud_flag", axis=1)

    # One-hot encoding (learn categories)
    X_encoded = pd.get_dummies(X, drop_first=True)

    # Save column order
    feature_columns = X_encoded.columns.tolist()
    joblib.dump(feature_columns, f"{ARTIFACT_DIR}/feature_columns.pkl")

    # Scale amount
    scaler = StandardScaler()
    X_encoded["amount_scaled"] = scaler.fit_transform(
        X_encoded[["amount (INR)"]]
    )

    X_encoded.drop(columns=["amount (INR)"], inplace=True)

    # Save scaler
    joblib.dump(scaler, f"{ARTIFACT_DIR}/scaler.pkl")

    print("Preprocessing artifacts saved:")
    print("- feature_columns.pkl")
    print("- scaler.pkl")


if __name__ == "__main__":
    fit_preprocessor("data/raw/transactions.csv")
