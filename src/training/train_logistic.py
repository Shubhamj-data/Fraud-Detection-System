import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

from src.preprocessing.preprocessor_batch import batch_transform


def train_logistic():
    # 1. Load raw data
    data = pd.read_csv("data/raw/transactions.csv")

    # 2. Separate target
    y = data["fraud_flag"]

    # 3. Batch preprocessing (FAST)
    X = batch_transform(data)

    # 4. Train-test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # 5. Apply SMOTE on training data only
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    # 6. Train Logistic Regression
    model = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        class_weight="balanced"
    )
    model.fit(X_train_res, y_train_res)

    # 7. Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- LOGISTIC REGRESSION (BASELINE) ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nROC-AUC:", roc_auc_score(y_test, y_proba))

    # 8. Save model
    joblib.dump(model, "artifacts/logistic_model.pkl")
    print("\nLogistic model saved to artifacts/logistic_model.pkl")

def train_logistic_retrain(X, y):
    """
    Train Logistic Regression on provided X, y
    Used for retraining pipeline
    """
    model = LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        class_weight="balanced"
    )
    model.fit(X, y)
    return model


if __name__ == "__main__":
    train_logistic()

