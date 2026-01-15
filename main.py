import sys
import os

sys.path.append(os.path.abspath("."))

print("\n🚀 FRAUD DETECTION PIPELINE STARTED\n")

# -----------------------------------
# STEP 1: Fetch + merge Google Sheets data
# -----------------------------------
print("📥 STEP 1: Fetching & merging data from Google Sheets...")
from src.preprocessing.retrain_from_sheets import main as retrain_from_sheets
retrain_from_sheets()

# -----------------------------------
# STEP 2 + 3: Preprocess + Train + Select model
# -----------------------------------
print("\n🤖 STEP 2 & 3: Preprocessing + Training...")
from src.preprocessing.train_and_select_model import main as train_and_select_model
train_and_select_model()

print("\n✅ PIPELINE COMPLETED SUCCESSFULLY\n")
