from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import shap

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
    Explicație pentru un singur produs.

    Include:
    - scorul baseline și eticheta deterministă
    - predicția și probabilitatea modelului ML
    - principalii factori SHAP care explică exclusiv predicția modelului ML,
      nu verdictul final de recomandare
    """
    ScorFinal: float
    Merita: int
    MeritaML: int
    ProbabilitateML: float
    TopFactoriML: list[FactorExplanation]


def _ensure_dataframe(product: dict[str, Any] | pd.Series | pd.DataFrame) -> pd.DataFrame:
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
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Lipsesc coloane necesare pentru explain_product: {missing}")


def _validate_numeric_columns(df: pd.DataFrame, cols: list[str]) -> None:
    for col in cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise ValueError(f"Coloana '{col}' trebuie să fie numerică")


def _clean_feature_name(feature_name: str) -> str:
    """
    Elimină prefixele generate de ColumnTransformer, de tip:
    'log_cols__n_of_reviews' -> 'n_of_reviews'
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
    Extrage top K factori care au contribuit cel mai mult la predicția
    modelului Logistic Regression pentru un produs dat.
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
            direction = "creste_probabilitatea"
        elif shap_value < 0:
            direction = "scade_probabilitatea"
        else:
            direction = "impact_neutru"

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


def _build_background_transformed(
    preprocessor,
    sample_size: int = 100,
    random_state: int = 42
):
    """
    Construiește un background dataset pentru SHAP pornind din datele
    proiectului și aplicând aceleași transformări ca în pipeline-ul ML.

    Acest background este extras din datasetul disponibil al proiectului și
    este folosit ca referință pentru explicarea locală a predicției modelului.
    """
    background_df = load_skincare_dv()
    background_df = add_engineered_features(background_df)

    missing = [col for col in MODEL_FEATURES if col not in background_df.columns]
    if missing:
        raise ValueError(f"Lipsesc coloane necesare pentru background SHAP: {missing}")

    if len(background_df) > sample_size:
        background_df = background_df.sample(n=sample_size, random_state=random_state)

    background_X = background_df[MODEL_FEATURES].copy()
    background_transformed = preprocessor.transform(background_X)

    return background_transformed


def explain_product(
    product: dict[str, Any] | pd.Series | pd.DataFrame,
    top_k: int = 3
) -> ProductExplanation:
    """
    Explicație pentru un singur produs.

    Include:
    - scorul baseline și eticheta deterministă
    - predicția și probabilitatea modelului ML
    - principalii factori SHAP care explică exclusiv predicția modelului ML

    Important:
    - factorii SHAP NU explică verdictul final de recomandare
    - factorii SHAP NU explică modulul de user matching
    - factorii SHAP NU explică scorul baseline
    """
    if top_k <= 0:
        raise ValueError("top_k trebuie să fie un număr pozitiv.")

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

    baseline_df = add_log_features(raw_product_df.copy())
    baseline_df = compute_score_with_scaler(baseline_df, score_scaler)
    baseline_df = label_with_threshold(baseline_df, threshold)

    scor_final = float(baseline_df.iloc[0]["ScorFinal"])
    merita = int(baseline_df.iloc[0]["Merita"])

    preprocessor = full_system.named_steps["preprocessor"]
    classifier = full_system.named_steps["classifier"]

    X_ml = ml_product_df.copy()
    merita_ml = int(full_system.predict(X_ml)[0])
    probabilitate_ml = float(full_system.predict_proba(X_ml)[0, 1])

    X_transformed = preprocessor.transform(X_ml)

    transformed_feature_names = list(preprocessor.get_feature_names_out())
    background_transformed = _build_background_transformed(preprocessor)

    explainer = shap.LinearExplainer(classifier, background_transformed)

    shap_values = explainer.shap_values(X_transformed)
    shap_values = np.asarray(shap_values)

    if shap_values.ndim == 1:
        shap_row = shap_values
    else:
        shap_row = shap_values[0]

    top_factori_ml = _extract_top_factors(
        shap_row=shap_row,
        feature_names=transformed_feature_names,
        input_row=X_ml,
        top_k=top_k
    )

    return ProductExplanation(
        ScorFinal=scor_final,
        Merita=merita,
        MeritaML=merita_ml,
        ProbabilitateML=probabilitate_ml,
        TopFactoriML=top_factori_ml
    )


def explanation_to_dict(explanation: ProductExplanation) -> dict[str, Any]:
    """
    Generează o explicație pentru un produs.

    Returnează:
    - ScorFinal și Merita pentru componenta baseline
    - MeritaML și ProbabilitateML pentru componenta ML
    - TopFactoriML, adică factorii SHAP care explică exclusiv predicția modelului ML

    Notă:
    Această funcție nu explică verdictul final de recommendation și nici
    componenta euristică de user matching.
    """
    return {
        "ScorFinal": explanation.ScorFinal,
        "Merita": explanation.Merita,
        "MeritaML": explanation.MeritaML,
        "ProbabilitateML": explanation.ProbabilitateML,
        "TopFactoriML": [
            {
                "feature": factor.feature,
                "feature_value": factor.feature_value,
                "shap_value": factor.shap_value,
                "impact_abs": factor.impact_abs,
                "direction": factor.direction,
            }
            for factor in explanation.TopFactoriML
        ],
    }


def print_explanation(explanation: ProductExplanation) -> None:
    print("=== EXPLICAȚIE PRODUS ===")
    print(f"ScorFinal: {explanation.ScorFinal:.4f}")
    print(f"Merita (baseline): {explanation.Merita}")
    print(f"MeritaML: {explanation.MeritaML}")
    print(f"ProbabilitateML: {explanation.ProbabilitateML:.4f}")
    print("Top factori ML (SHAP):")

    for i, factor in enumerate(explanation.TopFactoriML, start=1):
        print(
            f"{i}. {factor.feature} = {factor.feature_value} | "
            f"SHAP={factor.shap_value:.6f} | "
            f"{factor.direction}"
        )