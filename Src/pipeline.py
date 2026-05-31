from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from Src.inference import (
    load_bundle,
    inspect_baseline_input,
    prepare_baseline_dataframe,
    prepare_ml_dataframe,
    add_ml_predictions,
)
from Src.user_profile import UserProfile
from Src.user_matching import match_product_to_user
from Src.recommendation import build_final_recommendation


@dataclass
class FullPipelineResult:
    """
    Final output of the complete product evaluation pipeline, including:
    """
    FinalScore: float
    IsRecommended: int
    IsRecommendedML: int
    MLProbability: float
    FitScore: int
    IsCompatible: int
    FinalVerdict: str
    FinalExplanation: str
    PositiveSignals: list[str]
    NegativeSignals: list[str]


@dataclass
class PipelineResponse:
    """
    Structured response for the product evaluation pipeline, designed for backend/frontend communication."""
    status: str
    message: str
    missing_fields: list[str]
    invalid_fields: list[str]
    result: FullPipelineResult | None


def _normalize_product_input(product: dict[str, Any] | pd.Series) -> pd.DataFrame:
    if isinstance(product, dict):
        product_series = pd.Series(product)
    elif isinstance(product, pd.Series):
        product_series = product.copy()
    else:
        raise TypeError("product must be a dict or pandas.Series")

    return pd.DataFrame([product_series.to_dict()])


def evaluate_product_for_user(
    product: dict[str, Any] | pd.Series,
    user_profile: UserProfile
) -> PipelineResponse:
    """
    Runs the complete application flow:
    1. Baseline scoring (FinalScore + IsRecommended)
    2. ML classification (IsRecommendedML + MLProbability)
    3. User matching (FitScore + IsCompatible + PositiveSignals + NegativeSignals)
    4. Final recommendation verdict (FinalVerdict + FinalExplanation)
    """
    if not isinstance(user_profile, UserProfile):
        raise TypeError(
            f"user_profile must be an instance of UserProfile, not {type(user_profile).__name__}"
        )
    product_df = _normalize_product_input(product)
    baseline_report = inspect_baseline_input(product_df)

    missing_fields = sorted(
        set(baseline_report["missing_columns"] + baseline_report["missing_values"])
    )

    invalid_fields = sorted(
        set(
            baseline_report["non_numeric_fields"] +
            baseline_report["negative_count_fields"]
        )
    )

    if missing_fields:
        return PipelineResponse(
            status="insufficient_data",
            message=(
                "Product cannot be fully evaluated because required data is missing "
                "for the baseline component."
            ),
            missing_fields=missing_fields,
            invalid_fields=[],
            result=None,
        )

    if invalid_fields:
        return PipelineResponse(
            status="invalid_input",
            message=(
                "Product cannot be evaluated because some fields have invalid values "
            ),
            missing_fields=[],
            invalid_fields=invalid_fields,
            result=None,
        )

    bundle = load_bundle()

    baseline_df = prepare_baseline_dataframe(product_df, bundle)
    if baseline_df.empty:
        return PipelineResponse(
            status="insufficient_data",
            message=(
                "Product could not be fully evaluated because the baseline scoring step failed to produce a valid FinalScore. "
                "This may be due to missing or invalid input data that was not caught in the initial inspection."
            ),
            missing_fields=["FinalScore"],
            invalid_fields=[],
            result=None,
        )

    ml_df = prepare_ml_dataframe(baseline_df)
    full_df = add_ml_predictions(ml_df, bundle)

    if full_df.empty:
        return PipelineResponse(
            status="processing_error",
            message="Product could not be fully evaluated because the machine learning step failed to produce valid predictions. ",

            missing_fields=[],
            invalid_fields=[],
            result=None,
        )

    product_row = full_df.iloc[0]
    match_result = match_product_to_user(user_profile, product_row)

    final_result = build_final_recommendation(
        is_recommended=int(product_row["IsRecommended"]),
        is_recommended_ml=int(product_row["IsRecommendedML"]),
        is_compatible=match_result.IsCompatible
    )

    return PipelineResponse(
        status="ok",
        message="Product evaluation completed successfully.",
        missing_fields=[],
        invalid_fields=[],
        result=FullPipelineResult(
            FinalScore=float(product_row["FinalScore"]),
            IsRecommended=int(product_row["IsRecommended"]),
            IsRecommendedML=int(product_row["IsRecommendedML"]),
            MLProbability=float(product_row["MLProbability"]),
            FitScore=match_result.FitScore,
            IsCompatible=match_result.IsCompatible,
            FinalVerdict=final_result.verdict,
            FinalExplanation=final_result.explanation,
            PositiveSignals=match_result.PositiveSignals,
            NegativeSignals=match_result.NegativeSignals,
        ),
    )