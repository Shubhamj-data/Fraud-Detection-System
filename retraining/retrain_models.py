import sys, os
sys.path.append(os.path.abspath("."))

import pandas as pd
import joblib

from src.preprocessing.preprocessor_fit import fit_preprocessor
from src.preprocessing.preprocessor_batch import batch_transform

from src.training.train_logistic import train_logistic_retrain
from src.training.train_random_forest import train_random_forest_retrain
from src.training.train_xgboost import train_xgboost_retrain


DATA_PATH = "retraining/combined_data.csv"
TARGET_COL = "fraud_flag"

# STEP 1: Fit preprocessing artifacts
fit_preprocessor(DATA_PATH)

# STEP 2: Load data
df = pd.read_csv(DATA_PATH, low_memory=False)

# STEP 3: Separate target
y = df[TARGET_COL].values

# STEP 4: Transform features ONLY
X = batch_transform(df)

# STEP 4.1: Remove rows with NaNs (safe for retraining)
mask = ~X.isna().any(axis=1)
X = X[mask]
y = y[mask]

# STEP 5: Train models
log_model = train_logistic_retrain(X, y)
rf_model = train_random_forest_retrain(X, y)
xgb_model = train_xgboost_retrain(X, y)

# STEP 6: Save new models
joblib.dump(log_model, "artifacts/logistic_new.pkl")
joblib.dump(rf_model, "artifacts/random_forest_new.pkl")
joblib.dump(xgb_model, "artifacts/xgboost_new.pkl")

print("✅ Logistic, Random Forest, and XGBoost retrained successfully")
