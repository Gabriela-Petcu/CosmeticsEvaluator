from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "Data"
RAW_DIR = DATA_DIR / "Raw"
PROCESSED_DIR = DATA_DIR / "Processed"

RAW_SKINCARE_DV = RAW_DIR / "skincare_df.csv"
