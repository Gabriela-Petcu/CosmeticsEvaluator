import numpy as np
import pandas as pd


RAW_FEATURE_COLUMNS = [
    "n_of_reviews",
    "n_of_loves",
    "review_score",
    "price_per_ounce",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    missing = [col for col in RAW_FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Lipsesc coloane necesare pentru add_engineered_features: {missing}"
        )

    non_numeric = [
        col for col in RAW_FEATURE_COLUMNS
        if not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric:
        raise ValueError(
            f"Coloanele următoare trebuie să fie numerice (dtype numeric), "
            f"dar au fost găsite ca tip non-numeric: {non_numeric}. "
            f"Verifică dacă CSV-ul a fost citit corect."
        )

    out = df.copy()

    reviews = out["n_of_reviews"].fillna(0)
    loves = out["n_of_loves"].fillna(0)
    review_score = out["review_score"].fillna(0)
    price_per_ounce_clean = out["price_per_ounce"].fillna(0)

    out["popularity_score"] = np.log1p(reviews) + np.log1p(loves)
    out["engagement_score"] = loves / (reviews + 1)
    out["value_score"] = np.where(
        price_per_ounce_clean > 0,
        review_score / price_per_ounce_clean,
        0
    )
    out["review_strength"] = review_score * np.log1p(reviews)

    return out.replace([np.inf, -np.inf], 0).fillna(0)