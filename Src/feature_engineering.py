import numpy as np
import pandas as pd


RAW_FEATURE_COLUMNS = [
    "n_of_reviews",
    "n_of_loves",
    "review_score",
    "price_per_ounce",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add four derived features to the DataFrame based on raw product metrics.

    Engineered features:
    - popularity_score: log1p(n_of_reviews) + log1p(n_of_loves) (overall reach of the product)
    - engagement_score: n_of_loves / (n_of_reviews + 1) (how strongly users engage relative to review volume)
    - value_score: review_score / price_per_ounce (quality relative to price)
    - review_strength: review_score * log1p(n_of_reviews) (penalizes high-rated products with few reviews)
    """
    missing = [col for col in RAW_FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns for add_engineered_features: {missing}"
        )

    non_numeric = [
        col for col in RAW_FEATURE_COLUMNS
        if not pd.api.types.is_numeric_dtype(df[col])
    ]
    if non_numeric:
        raise ValueError(
            f"Columns must be numeric (dtype numeric), "
            f"but were found as non-numeric: {non_numeric}. "
            f"Verify that the CSV file was read correctly."
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