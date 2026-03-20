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
    ScorFinal: float
    Merita: int
    MeritaML: int
    ProbabilitateML: float
    FitScore: int
    SePotriveste: int
    VerdictFinal: str
    ExplicatieFinala: str
    MotivePozitive: list[str]
    MotiveNegative: list[str]


@dataclass
class PipelineResponse:
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
        raise TypeError("product trebuie să fie dict sau pandas.Series")

    return pd.DataFrame([product_series.to_dict()])


def evaluate_product_for_user(
    product: dict[str, Any] | pd.Series,
    user_profile: UserProfile
) -> PipelineResponse:
    """
    Rulează flow-ul complet al aplicației:

    1. baseline scoring
    2. clasificare ML
    3. user matching
    4. verdict final de recomandare

    Dacă produsul nu are date suficiente pentru baseline, funcția NU forțează
    o evaluare incompletă. În schimb, returnează un răspuns structurat,
    potrivit pentru backend/frontend.
    """
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
                "Produsul nu poate fi evaluat complet deoarece lipsesc date necesare "
                "pentru componenta baseline."
            ),
            missing_fields=missing_fields,
            invalid_fields=[],
            result=None,
        )

    if invalid_fields:
        return PipelineResponse(
            status="invalid_input",
            message=(
                "Produsul nu poate fi evaluat deoarece unele câmpuri necesare pentru "
                "baseline conțin valori invalide."
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
                "Produsul nu poate fi evaluat complet deoarece componenta baseline "
                "nu a putut calcula un scor final valid."
            ),
            missing_fields=["ScorFinal"],
            invalid_fields=[],
            result=None,
        )

    ml_df = prepare_ml_dataframe(baseline_df)
    full_df = add_ml_predictions(ml_df, bundle)

    if full_df.empty:
        return PipelineResponse(
            status="processing_error",
            message="Produsul nu a putut fi procesat complet pentru inferență.",
            missing_fields=[],
            invalid_fields=[],
            result=None,
        )

    product_row = full_df.iloc[0]

    match_result = match_product_to_user(user_profile, product_row)

    final_result = build_final_recommendation(
        merita=int(product_row["Merita"]),
        merita_ml=int(product_row["MeritaML"]),
        se_potriveste=match_result.SePotriveste
    )

    return PipelineResponse(
        status="ok",
        message="Evaluarea produsului a fost realizată cu succes.",
        missing_fields=[],
        invalid_fields=[],
        result=FullPipelineResult(
            ScorFinal=float(product_row["ScorFinal"]),
            Merita=int(product_row["Merita"]),
            MeritaML=int(product_row["MeritaML"]),
            ProbabilitateML=float(product_row["ProbabilitateML"]),
            FitScore=match_result.FitScore,
            SePotriveste=match_result.SePotriveste,
            VerdictFinal=final_result.verdict,
            ExplicatieFinala=final_result.explanation,
            MotivePozitive=match_result.PositiveSignals,
            MotiveNegative=match_result.NegativeSignals,
        ),
    )