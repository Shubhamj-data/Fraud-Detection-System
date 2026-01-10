import gspread
import pandas as pd
from google.oauth2.service_account import Credentials

SCOPES = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

SPREADSHEET_NAME = "Fraud_Detection_DB"
SHEET_NAME = "prediction_logs"


def fetch_data():
    creds = Credentials.from_service_account_file(
        "credentials/google_sheets_key.json",
        scopes=SCOPES
    )
    client = gspread.authorize(creds)
    sheet = client.open(SPREADSHEET_NAME).worksheet(SHEET_NAME)

    data = sheet.get_all_records()
    df = pd.DataFrame(data)

    df.to_csv("retraining/live_data.csv", index=False)
    print("✅ Live Google Sheets data saved")


if __name__ == "__main__":
    fetch_data()
