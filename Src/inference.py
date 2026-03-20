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
    Încarcă bundle-ul modelului salvat.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Nu există modelul la calea: {MODEL_PATH}. "
            "Rulează mai întâi training-ul."
        )

    bundle = joblib.load(MODEL_PATH)
    required_keys = ["full_system", "threshold", "score_scaler"]
    missing_keys = [k for k in required_keys if k not in bundle]
    if missing_keys:
        raise ValueError(f"Bundle invalid. Lipsesc cheile: {missing_keys}")

    return bundle


def inspect_baseline_input(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Verifică dacă datele necesare pentru baseline sunt disponibile și valide.

    Returnează un raport structurat cu:
    - missing_columns: coloane necesare care lipsesc complet
    - missing_values: coloane necesare care există, dar conțin valori lipsă
    - non_numeric_fields: coloane care există, dar nu pot fi interpretate numeric
    - negative_count_fields: count-uri invalide pentru log1p (valori negative)

    Acest raport este util pentru backend/frontend, pentru a explica
    de ce un produs nu poate primi evaluare completă.
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
    Adaugă log features, scorul baseline și eticheta Merita.

    Necesită ca datele de intrare să fie deja suficiente și valide pentru
    componenta baseline. Dacă există lipsuri sau valori invalide, acestea
    trebuie detectate anterior prin inspect_baseline_input().
    """
    threshold = float(bundle["threshold"])
    scaler = bundle["score_scaler"]

    df = add_log_features(df)
    df = compute_score_with_scaler(df, scaler)
    df = df.dropna(subset=["ScorFinal"]).copy()
    df = label_with_threshold(df, threshold)

    return df


def prepare_ml_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adaugă feature-urile necesare modelului ML și verifică existența lor.
    Valorile lipsă nu sunt eliminate aici, deoarece sunt tratate în
    pipeline-ul oficial al modelului.
    """
    out = df.copy()
    out = add_engineered_features(out)

    missing = [col for col in MODEL_FEATURES if col not in out.columns]
    if missing:
        raise ValueError(f"Lipsesc coloane necesare pentru ML: {missing}")

    return out


def add_ml_predictions(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """
    Adaugă coloanele MeritaML și ProbabilitateML peste un DataFrame deja pregătit.
    """
    full_system = bundle["full_system"]

    out = df.copy()
    missing = [col for col in MODEL_FEATURES if col not in out.columns]
    if missing:
        raise ValueError(f"Lipsesc coloane necesare pentru predicție ML: {missing}")

    out["MeritaML"] = full_system.predict(out[MODEL_FEATURES])
    out["ProbabilitateML"] = full_system.predict_proba(out[MODEL_FEATURES])[:, 1]

    return out


def build_baseline_ml_analysis_df(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Construiește DataFrame-ul complet de analiză, incluzând:
    - scorul baseline și eticheta Merita
    - feature engineering pentru ML
    - predicția modelului ML și probabilitatea asociată

    Se aplică doar produselor care au date suficiente pentru baseline.
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
    Încarcă datasetul brut și îl pregătește complet pentru inferență.
    """
    return build_baseline_ml_analysis_df()