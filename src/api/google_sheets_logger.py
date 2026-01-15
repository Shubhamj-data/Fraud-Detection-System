# src/api/google_sheets_logger.py

import os
import json
from datetime import datetime

import gspread
from google.oauth2.service_account import Credentials
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


def get_google_sheet():
    """
    Authenticate and return Google Sheet worksheet
    """
    service_account_json = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
    sheet_name = os.getenv("GOOGLE_SHEET_NAME")
    sheet_tab = os.getenv("GOOGLE_SHEET_DATA_TAB")

    if not service_account_json:
        raise ValueError("GOOGLE_SERVICE_ACCOUNT_JSON not found in .env")

    creds_dict = json.loads(service_account_json)

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]

    creds = Credentials.from_service_account_info(
        creds_dict,
        scopes=scopes
    )

    client = gspread.authorize(creds)

    sheet = client.open(sheet_name).worksheet(sheet_tab)
    return sheet


def log_transaction_to_sheet(input_data: dict, prediction: int, probability: float, model_version: str):
    """
    Append one prediction row to Google Sheets
    """
    try:
        sheet = get_google_sheet()

        row = [
            datetime.utcnow().isoformat(),

            input_data.get("transaction type"),
            input_data.get("merchant_category"),
            float(input_data.get("amount (INR)", 0)),

            input_data.get("transaction_status", "SUCCESS"),
            input_data.get("sender_age_group"),
            input_data.get("receiver_age_group"),

            input_data.get("sender_state"),
            input_data.get("sender_bank"),
            input_data.get("receiver_bank"),

            input_data.get("device_type"),
            input_data.get("network_type"),

            int(input_data.get("hour_of_day")),
            input_data.get("day_of_week"),
            int(input_data.get("is_weekend")),

            prediction,
            round(probability, 6),
            model_version,
        ]

        sheet.append_row(row, value_input_option="USER_ENTERED")

        print("✅ Google Sheets updated successfully")

    except Exception as e:
        print("❌ Google Sheets logging failed:", str(e))
