import pandas as pd
import joblib

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

from src.preprocessing.preprocessor_batch import batch_transform


def train_xgboost():
    # 1. Load raw data
    data = pd.read_csv("data/raw/transactions.csv")

    # 2. Separate target
    y = data["fraud_flag"]

    # 3. Batch preprocessing
    X = batch_transform(data)

    # 4. Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 5. Apply SMOTE on training data only
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 6. Train XGBoost
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train_res, y_train_res)

    # 7. Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- XGBOOST RESULTS ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nROC-AUC:", roc_auc_score(y_test, y_proba))

    # 8. Save model
    joblib.dump(model, "artifacts/xgboost_model.pkl")
    print("\nXGBoost model saved to artifacts/xgboost_model.pkl")


if __name__ == "__main__":
    train_xgboost()


def train_xgboost_retrain(X, y):
    """
    Retrain XGBoost on provided X, y
    """
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="auc",
        n_jobs=-1,
        random_state=42
    )
    model.fit(X, y)
    return model
