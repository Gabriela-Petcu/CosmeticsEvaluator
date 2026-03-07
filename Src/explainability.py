from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

from Src.scoring import add_log_features, compute_score_with_scaler, label_with_threshold


BUNDLE_PATH = Path("Models") / "bundle_v1.joblib"
MODEL_FEATURES = ["n_of_reviews", "n_of_loves", "review_score", "price_per_ounce"]


@dataclass
class FactorExplanation:
    feature: str
    feature_value: Any
    shap_value: float
    impact_abs: float
    direction: str


@dataclass
class ProductExplanation:
    ScorFinal: float
    Merita: int
    MeritaML: int
    ProbabilitateML: float
    TopFactori: list[FactorExplanation]


def _ensure_dataframe(product: dict[str, Any] | pd.Series | pd.DataFrame) -> pd.DataFrame:
    """
    Transformă inputul într-un DataFrame cu exact un rând.
    """
    if isinstance(product, dict):
        df = pd.DataFrame([product])
    elif isinstance(product, pd.Series):
        df = pd.DataFrame([product.to_dict()])
    elif isinstance(product, pd.DataFrame):
        df = product.copy()
    else:
        raise TypeError("product trebuie să fie dict, pandas.Series sau pandas.DataFrame")

    if len(df) != 1:
        raise ValueError("explain_product() acceptă exact un singur produs")

    return df


def _validate_required_columns(df: pd.DataFrame, required_cols: list[str]) -> None:
    """
    Verifică dacă există toate coloanele necesare.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Lipsesc coloane necesare pentru explain_product: {missing}")


def _load_bundle(bundle_path: str | Path = BUNDLE_PATH) -> dict[str, Any]:
    """
    Încarcă bundle-ul serializat.
    """
    bundle_path = Path(bundle_path)

    if not bundle_path.exists():
        raise FileNotFoundError(
            f"Nu există bundle-ul la calea: {bundle_path}. "
            "Rulează mai întâi training-ul pentru a genera Models/bundle_v1.joblib"
        )

    bundle = joblib.load(bundle_path)

    required_keys = ["full_system", "threshold", "score_scaler"]
    missing_keys = [k for k in required_keys if k not in bundle]
    if missing_keys:
        raise ValueError(f"Bundle invalid. Lipsesc cheile: {missing_keys}")

    return bundle


def _get_positive_class_shap_values(shap_values: Any) -> np.ndarray:
    """
    Normalizează diferitele formate SHAP într-un array 2D de forma:
    (n_samples, n_features) pentru clasa pozitivă.
    """
    if isinstance(shap_values, list):
        # format clasic pentru clasificare binară: [class_0, class_1]
        if len(shap_values) < 2:
            raise ValueError("SHAP a returnat o listă neașteptată pentru clasificare binară")
        return np.asarray(shap_values[1])

    shap_array = np.asarray(shap_values)

    if shap_array.ndim == 2:
        # deja (n_samples, n_features)
        return shap_array

    if shap_array.ndim == 3:
        # posibil (n_samples, n_features, n_classes)
        if shap_array.shape[2] < 2:
            raise ValueError("SHAP 3D array fără clasa pozitivă")
        return shap_array[:, :, 1]

    raise ValueError(f"Format SHAP neașteptat. ndim={shap_array.ndim}")


def _clean_feature_name(feature_name: str) -> str:
    """
    Elimină prefixele generate de ColumnTransformer:
    ex: log_cols__n_of_reviews -> n_of_reviews
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
    Extrage cei mai importanți top_k factori pe baza valorilor absolute SHAP.
    """
    factors: list[FactorExplanation] = []

    for raw_feature_name, shap_value in zip(feature_names, shap_row):
        clean_name = _clean_feature_name(raw_feature_name)

        if clean_name in input_row.columns:
            feature_value = input_row.iloc[0][clean_name]
        else:
            feature_value = None

        direction = "creste_probabilitatea" if shap_value > 0 else "scade_probabilitatea"

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
    bundle_path: str | Path = BUNDLE_PATH,
    top_k: int = 3
) -> ProductExplanation:
    """
    Explică decizia sistemului pentru un singur produs.

    Returnează:
    - ScorFinal (baseline)
    - Merita (baseline)
    - MeritaML
    - ProbabilitateML
    - TopFactori (SHAP)
    """
    # 1. Input -> DataFrame
    product_df = _ensure_dataframe(product)
    _validate_required_columns(product_df, MODEL_FEATURES)

    # păstrăm doar feature-urile relevante
    product_df = product_df[MODEL_FEATURES].copy()

    # 2. Încarcă bundle
    bundle = _load_bundle(bundle_path)
    full_system = bundle["full_system"]
    threshold = bundle["threshold"]
    score_scaler = bundle["score_scaler"]

    # 3. Baseline
    baseline_df = add_log_features(product_df.copy())
    baseline_df = compute_score_with_scaler(baseline_df, score_scaler)
    baseline_df = label_with_threshold(baseline_df, threshold)

    scor_final = float(baseline_df.iloc[0]["ScorFinal"])
    merita = int(baseline_df.iloc[0]["Merita"])

    # 4. ML predictions
    preprocessor = full_system.named_steps["preprocessor"]
    classifier = full_system.named_steps["classifier"]

    X_ml = product_df.copy()
    X_transformed = preprocessor.transform(X_ml)

    merita_ml = int(classifier.predict(X_transformed)[0])
    probabilitate_ml = float(classifier.predict_proba(X_transformed)[0, 1])

    # 5. Feature names după preprocessing
    transformed_feature_names = list(preprocessor.get_feature_names_out())

    # 6. SHAP
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_transformed)
    shap_values_positive = _get_positive_class_shap_values(shap_values)

    top_factori = _extract_top_factors(
        shap_row=shap_values_positive[0],
        feature_names=transformed_feature_names,
        input_row=X_ml,
        top_k=top_k
    )

    return ProductExplanation(
        ScorFinal=scor_final,
        Merita=merita,
        MeritaML=merita_ml,
        ProbabilitateML=probabilitate_ml,
        TopFactori=top_factori
    )


def explanation_to_dict(explanation: ProductExplanation) -> dict[str, Any]:
    """
    Convertește rezultatul într-un dict simplu, util pentru print, JSON sau API.
    """
    return {
        "ScorFinal": explanation.ScorFinal,
        "Merita": explanation.Merita,
        "MeritaML": explanation.MeritaML,
        "ProbabilitateML": explanation.ProbabilitateML,
        "TopFactori": [
            {
                "feature": factor.feature,
                "feature_value": factor.feature_value,
                "shap_value": factor.shap_value,
                "impact_abs": factor.impact_abs,
                "direction": factor.direction,
            }
            for factor in explanation.TopFactori
        ],
    }


def print_explanation(explanation: ProductExplanation) -> None:
    """
    Afișare prietenoasă în consolă.
    """
    print("=== EXPLICAȚIE PRODUS ===")
    print(f"ScorFinal: {explanation.ScorFinal:.4f}")
    print(f"Merita (baseline): {explanation.Merita}")
    print(f"MeritaML: {explanation.MeritaML}")
    print(f"ProbabilitateML: {explanation.ProbabilitateML:.4f}")
    print("Top factori:")

    for i, factor in enumerate(explanation.TopFactori, start=1):
        print(
            f"{i}. {factor.feature} = {factor.feature_value} | "
            f"SHAP={factor.shap_value:.6f} | "
            f"{factor.direction}"
        )