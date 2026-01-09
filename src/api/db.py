import sqlite3
import json
from datetime import datetime

DB_PATH = "predictions.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS prediction_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            transaction_json TEXT,
            fraud_prediction INTEGER,
            fraud_probability REAL,
            model_version TEXT
        )
    """)

    conn.commit()
    conn.close()


def log_prediction(input_data, prediction, probability, model_version):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO prediction_logs
        (timestamp, transaction_json, fraud_prediction, fraud_probability, model_version)
        VALUES (?, ?, ?, ?, ?)
    """, (
        datetime.utcnow().isoformat(),
        json.dumps(input_data),
        int(prediction),
        float(probability),
        model_version
    ))

    conn.commit()
    conn.close()
