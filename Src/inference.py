import joblib
import pandas as pd
from typing import Any

from Src.config import MODEL_PATH, MODEL_FEATURES
from Src.io import load_skincare_dv
from Src.scoring import add_log_features, compute_score_with_scaler, label_with_threshold


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


def prepare_baseline_dataframe(df: pd.DataFrame, bundle: dict[str, Any]) -> pd.DataFrame:
    """
    Adaugă log features, scor baseline și eticheta Merita.
    """
    threshold = float(bundle["threshold"])
    scaler = bundle["score_scaler"]

    df = add_log_features(df)
    df = compute_score_with_scaler(df, scaler)
    df = label_with_threshold(df, threshold)

    return df


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


def build_full_analysis_df(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Returnează datasetul complet pregătit pentru analiză:
    baseline + predicții ML.
    """
    if df is None:
        df = load_skincare_dv()

    bundle = load_bundle()
    df = prepare_baseline_dataframe(df, bundle)
    df = add_ml_predictions(df, bundle)

    return df


def load_and_prepare_dataset() -> pd.DataFrame:
    """
    Încarcă datasetul brut și îl pregătește complet pentru inferență.
    """
    return build_full_analysis_df()