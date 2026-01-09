from src.preprocessing.preprocessor_transform import transform_input

sample_input = {
    "transaction type": "P2P",
    "merchant_category": "Food",
    "amount (INR)": 500,
    "transaction_status": "SUCCESS",
    "sender_age_group": "26-35",
    "receiver_age_group": "18-25",
    "sender_state": "Maharashtra",
    "sender_bank": "HDFC",
    "receiver_bank": "ICICI",
    "device_type": "Android",
    "network_type": "4G",
    "hour_of_day": 14,
    "day_of_week": "Monday",
    "is_weekend": 0
}

X = transform_input(sample_input)
print(X.shape)
print(X.head())
