import numpy as np
import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adaugă feature-uri derivate pentru modelul ML.
    """
    out = df.copy()

    reviews = out["n_of_reviews"].fillna(0)
    loves = out["n_of_loves"].fillna(0)
    review_score = out["review_score"]
    price_per_ounce = out["price_per_ounce"]

    out["popularity_score"] = np.log1p(reviews) + np.log1p(loves)
    out["engagement_score"] = loves / (reviews + 1)

    safe_price = price_per_ounce.replace(0, np.nan)
    out["value_score"] = review_score / safe_price

    out["review_strength"] = review_score * np.log1p(reviews)
    out = out.replace([np.inf, -np.inf], np.nan)
    return out