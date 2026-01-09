REQUIRED_FEATURES = [
    "transaction type",
    "merchant_category",
    "amount (INR)",
    "transaction_status",
    "sender_age_group",
    "receiver_age_group",
    "sender_state",
    "sender_bank",
    "receiver_bank",
    "device_type",
    "network_type",
    "hour_of_day",
    "day_of_week",
    "is_weekend"
]

CATEGORICAL_FEATURES = [
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

NUMERICAL_FEATURES = [
    "amount (INR)",
    "hour_of_day",
    "is_weekend"
]


def validate_input(input_data: dict):
    """
    Validate incoming data for prediction.
    Ensures required fields exist and types are correct.
    """

    missing_fields = [
        feature for feature in REQUIRED_FEATURES
        if feature not in input_data
    ]

    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")

    # Type checks (basic safety)
    if not isinstance(input_data["amount (INR)"], (int, float)):
        raise ValueError("amount (INR) must be numeric")

    if not isinstance(input_data["hour_of_day"], int):
        raise ValueError("hour_of_day must be integer")

    if input_data["hour_of_day"] < 0 or input_data["hour_of_day"] > 23:
        raise ValueError("hour_of_day must be between 0 and 23")

    if input_data["is_weekend"] not in [0, 1]:
        raise ValueError("is_weekend must be 0 or 1")

    return True
