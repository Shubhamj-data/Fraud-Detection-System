import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from src.preprocessing.preprocessor import preprocess_data


ARTIFACT_DIR = "artifacts"


def train_and_evaluate(model, X_train, X_test, y_train, y_test, name):
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probs)
    print(f"📊 {name} ROC-AUC: {auc:.4f}")
    return auc, model


def main():
    print("🤖 Training & selecting best model...")

    # -----------------------------
    # Load preprocessed data
    # -----------------------------
    X, y = preprocess_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    results = {}

    # -----------------------------
    # Logistic Regression
    # -----------------------------
    log_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    auc, model = train_and_evaluate(
        log_model, X_train, X_test, y_train, y_test, "Logistic"
    )
    results["logistic"] = (auc, model)

    # -----------------------------
    # Random Forest
    # -----------------------------
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=20,
        random_state=42,
        n_jobs=-1
    )
    auc, model = train_and_evaluate(
        rf_model, X_train, X_test, y_train, y_test, "RandomForest"
    )
    results["random_forest"] = (auc, model)

    # -----------------------------
    # XGBoost
    # -----------------------------
    xgb_model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        random_state=42
    )
    auc, model = train_and_evaluate(
        xgb_model, X_train, X_test, y_train, y_test, "XGBoost"
    )
    results["xgboost"] = (auc, model)

    # -----------------------------
    # Select best model (ROC < 1.0)
    # -----------------------------
    filtered = {
        k: v for k, v in results.items() if v[0] < 0.999
    }

    best_name, (best_auc, best_model) = max(
        filtered.items(), key=lambda x: x[1][0]
    )

    joblib.dump(best_model, f"{ARTIFACT_DIR}/model.pkl")

    print(f"\n✅ BEST MODEL SELECTED: {best_name.upper()}")
    print(f"🏆 ROC-AUC: {best_auc:.4f}")
    print("📦 Saved as artifacts/model.pkl")

    return best_name, best_auc

