# Fraud Detection System (ML + Flask)

🚀 Live Application

🔗 Live App:
https://fraud-detection-system-u7cv.onrender.com

📌 Project Highlights

📊 Performed EDA & feature engineering on large-scale transaction data

🧠 Trained and compared Logistic Regression, Random Forest, and XGBoost

✅ Selected XGBoost for deployment based on generalization (ROC-AUC)

🌐 Built a Flask web app for real-time fraud prediction

🧾 Logged predictions to Google Sheets (online database)

🔁 Implemented retraining pipeline using live data

🔄 Maintained feature schema versioning to avoid inference mismatch

📈 Designed data flow compatible with Power BI dashboards

☁️ Deployed using Render (free tier) with GitHub CI/CD

🧱 System Architecture
User Input (Web Form / API)
        ↓
Flask Application
        ↓
Preprocessing Pipeline
        ↓
XGBoost Model (Production)
        ↓
Prediction Output
        ↓
Google Sheets (Live Database)
        ↓
Weekly Retraining Pipeline
        ↓
GitHub → Render Auto-Deploy

🛠️ Tech Stack

Languages & Libraries

Python

Pandas, NumPy

Scikit-learn

XGBoost

Imbalanced-learn (SMOTE)

Backend & Deployment

Flask

Render (Free Tier)

GitHub

Data & Storage

Google Sheets (as online database)

SQLite (local logging)

Visualization

Power BI (planned / optional extension)

📂 Project Structure
Fraud-Detection-System/
│
├── src/
│   ├── api/
│   │   ├── routes.py
│   │   ├── sheets_logger.py
│   │   └── db.py
│   │
│   ├── preprocessing/
│   │   ├── schema.py
│   │   ├── preprocessor_fit.py
│   │   ├── preprocessor_batch.py
│   │   └── preprocessor_transform.py
│   │
│   └── training/
│       ├── train_logistic.py
│       ├── train_random_forest.py
│       └── train_xgboost.py
│
├── retraining/
│   ├── fetch_from_google_sheets.py
│   ├── merge_data.py
│   ├── retrain_models.py
│   └── evaluate_models.py
│
├── artifacts/
│   ├── xgboost_model.pkl          # production model
│   ├── feature_columns.pkl        # feature schema
│   ├── scaler.pkl
│   └── unique_values.json
│
├── templates/
│   ├── index.html
│   └── history.html
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md

🧠 Model Training & Evaluation
Models Trained

Logistic Regression (baseline)

Random Forest

XGBoost

Evaluation Metric

ROC-AUC

Final Results (latest retraining)
Model	ROC-AUC
Logistic Regression	~0.60
Random Forest	~1.00 (overfitting risk)
XGBoost (Selected)	~0.99

✅ XGBoost selected for deployment due to better generalization.

🔁 Retraining Strategy

New prediction data stored in Google Sheets

Weekly retraining pipeline:

Fetch live data

Merge with historical data

Retrain all models

Evaluate performance

Promote best model

Update feature schema

Push to GitHub → auto-deploy on Render

⚠️ Model and feature_columns.pkl are always versioned together to prevent feature mismatch.

🌐 API Endpoints
Health Check
GET /health

JSON Prediction API
POST /predict


Payload Example

{
  "transaction type": "P2P",
  "merchant_category": "Food",
  "amount (INR)": 1200,
  "sender_state": "Maharashtra",
  ...
}

Web UI
GET /
POST /predict-form

🔐 Security & Best Practices

❌ No raw data committed

❌ No credentials in repo

✅ Secrets managed via environment variables

✅ Robust error handling

✅ Deployment-safe artifact loading

📊 Power BI Dashboard (Planned)

Fraud trend over time

Fraud probability distribution

State-wise fraud heatmap

Model version tracking

📌 Key Learnings

Handling training–inference feature drift

Managing class imbalance

Avoiding overfitting in fraud models

Deploying ML systems on free cloud resources

Designing retraining-ready pipelines

👤 Author

Shubham Jadhao
Aspiring Data Scientist | Machine Learning Engineer

🔗 GitHub: https://github.com/Shubhamj-data
