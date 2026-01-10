import pandas as pd

old_data = pd.read_csv("data/raw/transactions.csv")
new_data = pd.read_csv("retraining/new_live_data.csv")

combined = pd.concat([old_data, new_data], ignore_index=True)

combined.to_csv("retraining/combined_training_data.csv", index=False)

print("Combined dataset saved")
