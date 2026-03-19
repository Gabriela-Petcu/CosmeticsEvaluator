from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "Data"
RAW_DIR = DATA_DIR / "Raw"
PROCESSED_DIR = DATA_DIR / "Processed"

MODELS_DIR = PROJECT_ROOT / "Models"
MODEL_PATH = MODELS_DIR / "bundle_logreg_v1.joblib"

RAW_SKINCARE_DV = RAW_DIR / "skincare_df.csv"

MODEL_FEATURES = [
    "n_of_reviews",
    "n_of_loves",
    "review_score",
    "price_per_ounce",
    "popularity_score",
    "engagement_score",
    "value_score",
    "review_strength"
]

SCORE_COLUMNS = [
    "review_score",
    "log_reviews",
    "log_loves",
    "price_per_ounce"
]

LOG_FEATURE_COLUMNS = [
    "n_of_reviews",
    "n_of_loves"
]

STANDARD_FEATURE_COLUMNS = [
    "review_score",
    "price_per_ounce",
    "popularity_score",
    "engagement_score",
    "value_score",
    "review_strength"
]