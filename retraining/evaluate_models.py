import sys, os
sys.path.append(os.path.abspath("."))

import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score

from src.preprocessing.preprocessor_batch import batch_transform

DATA_PATH = "retraining/combined_data.csv"

# Load data
df = pd.read_csv(DATA_PATH, low_memory=False)

# Separate target
y = df["fraud_flag"].values

# Transform features
X = batch_transform(df)

# Remove NaN rows (same logic as retraining)
mask = ~X.isna().any(axis=1)
X = X[mask]
y = y[mask]

models = {
    "logistic": joblib.load("artifacts/logistic_new.pkl"),
    "random_forest": joblib.load("artifacts/random_forest_new.pkl"),
    "xgboost": joblib.load("artifacts/xgboost_new.pkl"),
}

print("\n--- MODEL COMPARISON (ROC-AUC) ---")
scores = {}

for name, model in models.items():
    auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
    scores[name] = auc
    print(f"{name}: {auc:.4f}")

best_model = max(scores, key=scores.get)
print(f"\n✅ Best model based on ROC-AUC: {best_model}")
