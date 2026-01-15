import pandas as pd
import json

# Load dataset (use the same path you used in EDA)
df = pd.read_csv("data/raw/transactions.csv")

categorical_columns = [
    "transaction type",
    "merchant_category",
    "transaction_status",
    "sender_age_group",
    "receiver_age_group",
    "sender_state",
    "sender_bank",
    "receiver_bank",
    "device_type",
    "network_type",
    "day_of_week"
]

unique_values = {}

for col in categorical_columns:
    unique_values[col] = sorted(df[col].dropna().unique().tolist())

# Save as JSON (this will drive UI + validation)
with open("artifacts/unique_values.json", "w") as f:
    json.dump(unique_values, f, indent=4)

print("✅ Unique values extracted and saved to artifacts/unique_values.json")
