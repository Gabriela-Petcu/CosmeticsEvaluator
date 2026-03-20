from pathlib import Path

import pandas as pd

from .config import RAW_SKINCARE_DV


TEXT_COLUMNS_TO_CLEAN = ["brand", "name"]


def _fix_mojibake_text(value: object) -> object:
    """
    Încearcă să repare texte de tip mojibake, de exemplu:
    'Cr√®me Ancienne¬Æ' -> 'Crème Ancienne®'
    """
    if pd.isna(value):
        return value

    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return text

    suspicious_markers = ["√", "¬", "Ã", "Â", "Ð", "Ñ"]
    if not any(marker in text for marker in suspicious_markers):
        return text

    attempts = [
        ("mac_roman", "utf-8"),
        ("latin1", "utf-8"),
        ("cp1252", "utf-8"),
    ]

    for source_encoding, target_encoding in attempts:
        try:
            fixed = text.encode(source_encoding).decode(target_encoding)
            return fixed
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue

    return text


def _clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    for col in TEXT_COLUMNS_TO_CLEAN:
        if col in out.columns:
            out[col] = out[col].apply(_fix_mojibake_text)

    return out


def load_skincare_dv(path: str | Path = RAW_SKINCARE_DV) -> pd.DataFrame:
    """
    Încarcă dataset-ul skincare_df.csv într-un DataFrame și curăță
    eventualele probleme de encoding de pe coloanele text relevante.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Nu există fișierul dataset la calea: {path}")

    df = pd.read_csv(path)
    df = _clean_text_columns(df)

    return df