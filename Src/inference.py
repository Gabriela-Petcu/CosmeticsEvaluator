import joblib
import pandas as pd
from typing import Any

from Src.config import MODEL_PATH, MODEL_FEATURES
from Src.io import load_skincare_dv
from Src.scoring import add_log_features, compute_score_with_scaler, label_with_threshold
from Src.feature_engineering import add_engineered_features

BASELINE_REQUIRED_COLUMNS = [
    "n_of_reviews",
    "n_of_loves",
    "review_score",
    "price_per_ounce",
]


def load_bundle() -> dict[str, Any]:
    """
    Load the saved bundle from disk.
    Returns: 
    dict: dictionary containing at least the keys: 'full_system', 'threshold', 'score_scaler'
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No model found at {MODEL_PATH}. "
            "Run the training script first."
        )

    bundle = joblib.load(MODEL_PATH)
    required_keys = ["full_system", "threshold", "score_scaler"]
    missing_keys = [k for k in required_keys if k not in bundle]
    if missing_keys:
        raise ValueError(f"Bundle invalid. Missing keys: {missing_keys}")

    return bundle


def inspect_baseline_input(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Check if the input DataFrame has the necessary columns and valid data for the baseline scoring.
    
    Returns a structured report with four categories of issues:
    - missing_columns     : required columns that are entirely absent
    - missing_values      : required columns that exist but contain NaN
    - non_numeric_fields  : columns that exist but cannot be parsed as numeric
    - negative_count_fields : count columns with negative values (invalid for log1p)
    Parameters:
    df : pd.DataFrame
        Input DataFrame to inspect.
    Returns:
    dict[str, list[str]]
        Structured report with the four issue categories listed above.
    """
    missing_columns = [col for col in BASELINE_REQUIRED_COLUMNS if col not in df.columns]

    missing_values: list[str] = []
    non_numeric_fields: list[str] = []
    negative_count_fields: list[str] = []

    for col in BASELINE_REQUIRED_COLUMNS:
        if col not in df.columns:
            continue

        if df[col].isna().any():
            missing_values.append(col)
            continue

        numeric_series = pd.to_numeric(df[col], errors="coerce")

        if numeric_series.isna().any():
            non_numeric_fields.append(col)
            continue

        if col in ("n_of_reviews", "n_of_loves") and (numeric_series < 0).any():
            negative_count_fields.append(col)

    return {
        "missing_columns": missing_columns,
        "missing_values": missing_values,
        "non_numeric_fields": non_numeric_fields,
        "negative_count_fields": negative_count_fields,
    }


def prepare_baseline_dataframe(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """
    Add log features, the baseline score and the IsRecommended label.
    Parameters
    df : pd.DataFrame
        Input DataFrame containing at least the columns required for baseline scoring.
    bundle : dict[str, Any] 
        Loaded model bundle containing 'threshold' and 'score_scaler'.
    Returns
    pd.DataFrame
        Copy of the input DataFrame with og features, FinalScore, and IsRecommended added.
    """
    threshold = float(bundle["threshold"])
    scaler = bundle["score_scaler"]

    df = add_log_features(df)
    df = compute_score_with_scaler(df, scaler)
    df = df.dropna(subset=["FinalScore"]).copy()
    df = label_with_threshold(df, threshold)

    return df


def prepare_ml_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the necessary features for the ML model and check their existence.
    Parameters
    df : pd.DataFrame
        Input DataFrame that should already contain the baseline features and scores.
    Returns
    pd.DataFrame
        Copy of the input DataFrame with engineered features added.
    """
    out = df.copy()
    out = add_engineered_features(out)

    missing = [col for col in MODEL_FEATURES if col not in out.columns]
    if missing:
        raise ValueError(f"Missing columns required for ML features: {missing}")

    return out


def add_ml_predictions(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """
    Add IsRecommendedML and MLProbability columns to an already-prepared DataFrame.
    Parameters
    df : pd.DataFrame
        Input DataFrame that should already contain the ML features.
    bundle : dict[str, Any]
        Loaded model bundle containing the 'full_system' model.
    Returns
    pd.DataFrame
        Copy of the input DataFrame with 'IsRecommendedML' and 'MLProbability' added.
    """
    full_system = bundle["full_system"]

    out = df.copy()
    missing = [col for col in MODEL_FEATURES if col not in out.columns]
    if missing:
        raise ValueError(f"Missing columns required for ML predictions: {missing}")

    out["IsRecommendedML"] = full_system.predict(out[MODEL_FEATURES])
    out["MLProbability"] = full_system.predict_proba(out[MODEL_FEATURES])[:, 1]

    return out



def build_baseline_ml_analysis_df(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Build the complete analysis DataFrame, including:
    - baseline FinalScore and IsRecommended label
    - ML feature engineering
    - ML model prediction and associated probability
    Parameters
    df : pd.DataFrame | None
        Optional input DataFrame. If None, the raw skincare dataset will be loaded.
    Returns
    pd.DataFrame
         Fully scored and predicted DataFrame.
"""
    if df is None:
        df = load_skincare_dv()

    bundle = load_bundle()

    baseline_df = prepare_baseline_dataframe(df, bundle)
    ml_df = prepare_ml_dataframe(baseline_df)
    full_df = add_ml_predictions(ml_df, bundle)

    return full_df


def load_and_prepare_dataset() -> pd.DataFrame:
    """
    Load the raw dataset and prepare it fully for inference.
    Returns
    pd.DataFrame
        Fully prepared DataFrame ready for analysis and prediction.
    """
    return build_baseline_ml_analysis_df()