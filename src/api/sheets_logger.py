import gspread
from google.oauth2.service_account import Credentials
from datetime import datetime

import os
import json
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

# Spreadsheet & sheet names
SPREADSHEET_NAME = "Fraud_Detection_DB"
SHEET_NAME = "prediction_logs"


def get_sheet():
    creds_dict = json.loads(os.environ["GOOGLE_SHEETS_CREDENTIALS"])
    creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)

    client = gspread.authorize(creds)
    sheet = client.open("Fraud_Detection_DB").worksheet("prediction_logs")
    return sheet



def log_to_google_sheets(input_data, prediction, probability, model_version):
    try:
        sheet = get_sheet()

        row = [
            datetime.utcnow().isoformat(),
            input_data.get("transaction type"),
            input_data.get("merchant_category"),
            input_data.get("amount (INR)"),
            input_data.get("transaction_status"),
            input_data.get("sender_age_group"),
            input_data.get("receiver_age_group"),
            input_data.get("sender_state"),
            input_data.get("sender_bank"),
            input_data.get("receiver_bank"),
            input_data.get("device_type"),
            input_data.get("network_type"),
            input_data.get("hour_of_day"),
            input_data.get("day_of_week"),
            input_data.get("is_weekend"),
            prediction,
            probability,
            model_version
        ]

        sheet.append_row(row, value_input_option="USER_ENTERED")
        print("✅ Data successfully written to Google Sheets")

    except Exception as e:
        print("❌ Google Sheets logging failed:", e)
