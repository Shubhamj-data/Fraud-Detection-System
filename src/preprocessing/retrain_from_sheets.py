import os
import json
import pandas as pd
from dotenv import load_dotenv
import gspread
from google.oauth2.service_account import Credentials

load_dotenv()

SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
SHEET_NAME = os.getenv("GOOGLE_SHEET_NAME")          # Spreadsheet name
SHEET_DATA_TAB = os.getenv("GOOGLE_SHEET_DATA_TAB")  # Worksheet/tab name

CREDS_JSON = os.getenv("GOOGLE_SHEETS_CREDENTIALS")

OUTPUT_PATH = "data/processed/training_data.csv"
OLD_DATA_PATH = "data/processed/training_data.csv"



def connect_sheet():
    if not CREDS_JSON:
        raise ValueError("❌ GOOGLE_SHEETS_CREDENTIALS not found in .env")

    creds_dict = json.loads(CREDS_JSON)

    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    credentials = Credentials.from_service_account_info(
        creds_dict,
        scopes=scopes
    )

    return gspread.authorize(credentials)


def fetch_sheet_data():
    gc = connect_sheet()

    sheet = gc.open_by_key(SHEET_ID).worksheet(SHEET_DATA_TAB)
    records = sheet.get_all_records()

    if not records:
        print("⚠️ No new data found in Google Sheets")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    return df


def main():
    print("📥 STEP 1: Fetching & merging data from Google Sheets...")

    # -----------------------------
    # FIRST RUN SEED LOGIC
    # -----------------------------
    if not os.path.exists("data/processed/training_data.csv"):
        print("📥 First run detected — seeding from raw data")
        base_df = pd.read_csv("data/raw/transactions.csv")
    else:
        print("📥 Loading existing processed training data")
        base_df = pd.read_csv("data/processed/training_data.csv")

    # -----------------------------
    # Fetch Google Sheets data
    # -----------------------------
    sheet_df = fetch_sheet_data()

    if sheet_df.empty:
        print("⚠️ No new Google Sheets data found")
        combined_df = base_df
    else:
        print(f"➕ Merging {len(sheet_df)} new rows from Google Sheets")
        combined_df = pd.concat([base_df, sheet_df], ignore_index=True)

    # -----------------------------
    # Save back to processed
    # -----------------------------
    combined_df.to_csv("data/processed/training_data.csv", index=False)
    print("✅ Combined data saved to data/processed/training_data.csv")


if __name__ == "__main__":
    main()
