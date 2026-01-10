import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE

from src.preprocessing.preprocessor_batch import batch_transform


def train_random_forest():
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

    # 6. Train Random Forest
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    )

    model.fit(X_train_res, y_train_res)

    # 7. Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("\n--- RANDOM FOREST RESULTS ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nROC-AUC:", roc_auc_score(y_test, y_proba))

    # 8. Save model
    joblib.dump(model, "artifacts/random_forest_model.pkl")
    print("\nRandom Forest model saved to artifacts/random_forest_model.pkl")


if __name__ == "__main__":
    train_random_forest()

def train_random_forest_retrain(X, y):
    """
    Retrain Random Forest on provided X, y
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=10,
        min_samples_leaf=5,
        n_jobs=-1,
        random_state=42
    )
    model.fit(X, y)
    return model
