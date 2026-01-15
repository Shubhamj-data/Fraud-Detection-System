import os, json, gspread
from dotenv import load_dotenv
from google.oauth2.service_account import Credentials

load_dotenv()

creds = json.loads(os.getenv("GOOGLE_SHEETS_CREDENTIALS"))
scopes = [
    "https://www.googleapis.com/auth/spreadsheets",
    "https://www.googleapis.com/auth/drive"
]

credentials = Credentials.from_service_account_info(creds, scopes=scopes)
gc = gspread.authorize(credentials)

sheet = gc.open_by_key(os.getenv("GOOGLE_SHEET_ID"))
print("✅ Sheet found:", sheet.title)

ws = sheet.worksheet(os.getenv("GOOGLE_SHEET_NAME"))
print("✅ Worksheet found:", ws.title)
