import pandas as pd

old_data = pd.read_csv("data/raw/transactions.csv")
live_data = pd.read_csv("retraining/live_data.csv")

combined = pd.concat([old_data, live_data], ignore_index=True)

combined.to_csv("retraining/combined_data.csv", index=False)

print("✅ Combined training data created")
