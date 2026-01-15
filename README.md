# 🛡️ Fraud Detection System — End-to-End ML Project

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-Web_App-black?logo=flask)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Deployment](https://img.shields.io/badge/Deployed-Render-success)
![Database](https://img.shields.io/badge/Database-Google_Sheets-green)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

An **end-to-end Fraud Detection System** built using **Machine Learning, Flask, Google Sheets, and Power BI**, covering the complete ML lifecycle — from EDA and preprocessing to deployment and automated retraining — using **only free resources**.

---

## 🚀 Live Application
🔗 **Live URL:**  
https://fraud-detection-system-u7cv.onrender.com

---

## ✨ Key Features

- 📊 Exploratory Data Analysis & Feature Engineering  
- ⚖️ Severe class imbalance handling using SMOTE  
- 🧠 Trained and compared multiple ML models  
- ✅ XGBoost selected for production using ROC-AUC  
- 🌐 Flask web application (UI + REST API)  
- 🧾 Predictions logged to Google Sheets (online database)  
- 🔁 Automated retraining pipeline using live data  
- 🧩 Feature schema versioning to avoid inference mismatch  
- ☁️ Free cloud deployment with CI/CD  

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    A["User Input - Web or API"] --> B["Flask Application"]
    B --> C["Preprocessing Pipeline"]
    C --> D["XGBoost Model"]
    D --> E["Prediction Output"]
    D --> F["Google Sheets Database"]
    F --> G["Weekly Retraining Pipeline"]
    G --> H["GitHub Repository"]
    H --> I["Render Deployment"]
