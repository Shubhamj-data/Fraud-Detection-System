import sqlite3
import json
import pandas as pd

DB_PATH = "predictions.db"

def export_data():
    conn = sqlite3.connect(DB_PATH)

    df = pd.read_sql("""
        SELECT transaction_json, fraud_prediction
        FROM prediction_logs
    """, conn)

    conn.close()

    # Convert JSON column into structured columns
    records = []
    for _, row in df.iterrows():
        data = json.loads(row["transaction_json"])
        data["fraud_flag"] = row["fraud_prediction"]
        records.append(data)

    final_df = pd.DataFrame(records)
    final_df.to_csv("retraining/new_live_data.csv", index=False)

    print("Live data exported to retraining/new_live_data.csv")

if __name__ == "__main__":
    export_data()
