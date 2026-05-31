from pathlib import Path
import pandas as pd
from .config import RAW_SKINCARE_DV

TEXT_COLUMNS_TO_CLEAN = ["brand", "name"]


def _fix_mojibake_text(value: object) -> object:
    if pd.isna(value):
        return value
    if not isinstance(value, str):
        return value

    text = value.strip()
    if not text:
        return text

    replacements = [
        ("\u00ac\u00c6", "\u00ae"),
        (",\u00d1\u00a2", "\u00ae"),
        ("\u201a\u00d1\u00a2", "\u2122"),
        ("\u201a\u00c4\u00ec", "\u2014"),
        ("\u201a\u00c4\u00f4", "\u2019"),
        ("\u201a\u00c4\u00fa", "\u201c"),
        ("\u201a\u00c4\u00f9", "\u201d"),
        ("\u221a\u00ae", "\u00e8"),
        ("\u221a\u00a9", "\u00e9"),
        ("\u221a\u00e0", "\u00e0"),
        ("\u221a\u00fc", "\u00df"),
        ("R\u2248\u00e7", "R\u00e9"),
        ("\u2248\u00e7", "\u00e9"),
        ("\u00d1\u00a2", "\u2122"),
        ("\u00c3\u00a9", "\u00e9"),
        ("\u00c3\u00a8", "\u00e8"),
        ("\u00c3\u00a0", "\u00e0"),
        ("\u00e2\u20ac\u2122", "\u2019"),
        ("\u00e2\u20ac\u201d", "\u2014"),
        ("\u00e2\u20ac\u0153", "\u201c"),
        ("\u00e2\u20ac", "\u201d"),
    ]

    for bad, good in replacements:
        text = text.replace(bad, good)

    suspicious_markers = ["\u221a", "\u00ac", "\u00c3", "\u00c2", "\u00d0", "\u00d1", "\u201a", "\u2248"]
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
            if not any(marker in fixed for marker in suspicious_markers):
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
    Loads the skincare dataset from a CSV file and cleans text columns.
    """
    path = Path(path)
    requested_path = Path(path).resolve()
    base_data_path = Path(RAW_SKINCARE_DV).parent.resolve()

    if not requested_path.is_relative_to(base_data_path.parent):
        raise PermissionError("Unauthorized access to the filesystem.")
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    df = pd.read_csv(path)
    df = _clean_text_columns(df)

    return df