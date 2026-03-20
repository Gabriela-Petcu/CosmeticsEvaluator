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
    Adaugă feature-uri derivate pentru modelul ML.

    Observații:
    - valorile lipsă din n_of_reviews și n_of_loves sunt tratate ca 0,
      deoarece reprezintă count-uri
    - review_score și price_per_ounce nu sunt imputate aici; dacă lipsesc,
      unele feature-uri derivate pot deveni NaN și vor fi tratate ulterior
      în flow-ul de inferență / pipeline-ul ML
    """
    missing = [col for col in RAW_FEATURE_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"Lipsesc coloane necesare pentru add_engineered_features: {missing}"
        )

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