from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap
import threading

from Src.config import MODEL_FEATURES
from Src.inference import load_bundle
from Src.scoring import add_log_features, compute_score_with_scaler, label_with_threshold
from Src.io import load_skincare_dv
from Src.feature_engineering import add_engineered_features

RAW_REQUIRED_COLUMNS = [
    "n_of_reviews",
    "n_of_loves",
    "review_score",
    "price_per_ounce",
]

_SHAP_BACKGROUND_CACHE = None
_SHAP_CACHE_LOCK = threading.Lock()

@dataclass
class FactorExplanation:
    feature: str 
    feature_value: Any 
    shap_value: float 
    impact_abs: float 
    direction: str


@dataclass
class ProductExplanation:
    """
    Explanation for a single product.
 
    Includes:
    - baseline score and deterministic label
    - ML model prediction and probability
    - top SHAP factors explaining the ML model prediction only,
      not the final recommendation verdict
    """
    FinalScore: float
    IsRecommended: int
    IsRecommendedML: int
    MLProbability: float
    TopMLFactors: list[FactorExplanation]

def _get_background_data(preprocessor): 
    """
    Load and cache the background data for SHAP explanations.
    Parameters
    preprocessor : sklearn transformer
        The fitted preprocessor step from the full pipeline.
    Returns
    np.ndarray
        Transformed background data used to initialize the SHAP explainer."""
    global _SHAP_BACKGROUND_CACHE

    if _SHAP_BACKGROUND_CACHE is not None:
        return _SHAP_BACKGROUND_CACHE

    with _SHAP_CACHE_LOCK:
        if _SHAP_BACKGROUND_CACHE is None:
            bg_df = load_skincare_dv()
            bg_df = add_engineered_features(bg_df)
            sample = bg_df[MODEL_FEATURES].sample(n=100, random_state=42)
            _SHAP_BACKGROUND_CACHE = preprocessor.transform(sample)

    return _SHAP_BACKGROUND_CACHE

def _ensure_dataframe(product: dict[str, Any] | pd.Series | pd.DataFrame) -> pd.DataFrame:
    """
    Ensure the input product is in DataFrame format with exactly one row.
    Parameters
    product : dict[str, Any] | pd.Series | pd.DataFrame
        The input product data, which can be a dictionary, a pandas Series, or a single row DataFrame.
    Returns
    pd.DataFrame
         A DataFrame with exactly one row representing the product.
    """
    if isinstance(product, dict):
        df = pd.DataFrame([product])
    elif isinstance(product, pd.Series):
        df = pd.DataFrame([product.to_dict()])
    elif isinstance(product, pd.DataFrame):
        df = product.copy()
    else:
        raise TypeError("product must be a dict, pandas Series, or pandas DataFrame")

    if len(df) != 1:
        raise ValueError("explain_product() accepts exactly one product at a time")

    return df


def _validate_required_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for explain_product: {missing}")


def _validate_numeric_columns(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Column '{col}' must be numeric")


def _clean_feature_name(feature_name: str) -> str:
    """
    Strip ColumnTransformer prefixes from a feature name.
    Example: 'log_cols__n_of_reviews' -> 'n_of_reviews'
    """
    if "__" in feature_name:
        return feature_name.split("__", 1)[1]
    return feature_name


def _extract_top_factors(
    shap_row: np.ndarray,
    feature_names: list[str],
    input_row: pd.DataFrame,
    top_k: int = 3
) -> list[FactorExplanation]:
    """
    Extract the top K factors that contributed the most to the Logistic Regression model's prediction for a given product.
    
    Parameters
    shap_row : np.ndarray
        1-D array of SHAP values for a single product (one value per feature).
    feature_names : list[str]
        Feature names in the same order as shap_row (from get_feature_names_out()).
    input_row : pd.DataFrame
        Single-row DataFrame with the original (pre-transform) feature values.
    top_k : int, optional
        Number of top factors to return (default: 3).

    Returns
    list[FactorExplanation]
        Top-K FactorExplanation objects sorted by descending absolute SHAP value.
    """
    factors: list[FactorExplanation] = []

    for raw_feature_name, shap_value in zip(feature_names, shap_row):
        clean_name = _clean_feature_name(raw_feature_name)

        if clean_name in input_row.columns:
            raw_value = input_row.iloc[0][clean_name]
            feature_value = None if pd.isna(raw_value) else raw_value
        else:
            feature_value = None

        if shap_value > 0:
            direction = "increases_probability"
        elif shap_value < 0:
            direction = "decreases_probability"
        else:
            direction = "neutral_impact"

        factors.append(
            FactorExplanation(
                feature=clean_name,
                feature_value=feature_value,
                shap_value=float(shap_value),
                impact_abs=float(abs(shap_value)),
                direction=direction,
            )
        )

    factors.sort(key=lambda x: x.impact_abs, reverse=True)
    return factors[:top_k]


def explain_product(
    product: dict[str, Any] | pd.Series | pd.DataFrame,
    top_k: int = 3
) -> ProductExplanation:
    """
    Compute a full explanation for a single product, combining baseline scoring
    and ML prediction with SHAP feature attribution.

    Parameters
    product : dict, pd.Series, or pd.DataFrame
        Product data containing at least the RAW_REQUIRED_COLUMNS.
    top_k : int, optional
        Number of top SHAP factors to include in the explanation (default: 3).

    Returns
    ProductExplanation
        Dataclass with FinalScore, IsRecommended, IsRecommendedML,
        MLProbability, and TopMLFactors."""
    if top_k <= 0:
        raise ValueError("top_k must be a positive integer.")

    product_df = _ensure_dataframe(product)

    _validate_required_columns(product_df, RAW_REQUIRED_COLUMNS)
    _validate_numeric_columns(product_df, RAW_REQUIRED_COLUMNS)

    raw_product_df = product_df.copy()
    engineered_product_df = add_engineered_features(product_df.copy())

    _validate_required_columns(engineered_product_df, MODEL_FEATURES)
    _validate_numeric_columns(engineered_product_df, MODEL_FEATURES)

    ml_product_df = engineered_product_df[MODEL_FEATURES].copy()

    bundle = load_bundle()
    full_system = bundle["full_system"]
    threshold = float(bundle["threshold"])
    score_scaler = bundle["score_scaler"]

    preprocessor = full_system.named_steps["preprocessor"]
    classifier = full_system.named_steps["classifier"]
    baseline_df = add_log_features(raw_product_df.copy())
    baseline_df = compute_score_with_scaler(baseline_df, score_scaler)
    baseline_df = label_with_threshold(baseline_df, threshold)

    final_score = float(baseline_df.iloc[0]["FinalScore"])
    is_recommended = int(baseline_df.iloc[0]["IsRecommended"])

    X_ml = ml_product_df.copy()
    is_recommended_ml = int(full_system.predict(X_ml)[0])
    ml_probability = float(full_system.predict_proba(X_ml)[0, 1])

    X_transformed = preprocessor.transform(X_ml)
    transformed_feature_names = list(preprocessor.get_feature_names_out())

    background_transformed = _get_background_data(preprocessor)
    explainer = shap.LinearExplainer(classifier, background_transformed)

    shap_values = np.asarray(explainer.shap_values(X_transformed))
    shap_row = shap_values if shap_values.ndim == 1 else shap_values[0]

    top_ml_factors = _extract_top_factors(
        shap_row=shap_row,
        feature_names=transformed_feature_names,
        input_row=X_ml,
        top_k=top_k
    )

    return ProductExplanation(
        FinalScore=final_score,
        IsRecommended=is_recommended,
        IsRecommendedML=is_recommended_ml,
        MLProbability=ml_probability,
        TopMLFactors=top_ml_factors,
    )


def explanation_to_dict(explanation: ProductExplanation) -> dict[str, Any]:
    """
    Generates an explanation for a product.
    """
    return {
        "FinalScore": explanation.FinalScore,
        "IsRecommended": explanation.IsRecommended,
        "IsRecommendedML": explanation.IsRecommendedML,
        "MLProbability": explanation.MLProbability,
        "TopMLFactors": [
            {
                "feature": factor.feature,
                "feature_value": factor.feature_value,
                "shap_value": factor.shap_value,
                "impact_abs": factor.impact_abs,
                "direction": factor.direction,
            }
            for factor in explanation.TopMLFactors
        ],
    }


def print_explanation(explanation: ProductExplanation) -> None:
    print("PRODUCT EXPLANATION")
    print(f"FinalScore: {explanation.FinalScore:.4f}")
    print(f"IsRecommended (baseline): {explanation.IsRecommended}")
    print(f"IsRecommendedML: {explanation.IsRecommendedML}")
    print(f"MLProbability: {explanation.MLProbability:.4f}")
    print("Top ML Factors (SHAP):")

    for i, factor in enumerate(explanation.TopMLFactors, start=1):
        print(
            f"{i}. {factor.feature} = {factor.feature_value} | "
            f"SHAP={factor.shap_value:.6f} | "
            f"{factor.direction}"
        )