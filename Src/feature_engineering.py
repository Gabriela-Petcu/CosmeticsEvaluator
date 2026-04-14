import numpy as np
import pandas as pd


RAW_FEATURE_COLUMNS = [
    "n_of_reviews",
    "n_of_loves",
    "review_score",
    "price_per_ounce",
]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    # ... (verificare missing columns existentă) ...
    out = df.copy()
    
    # Pregătim datele pentru calcule sigure
    reviews = out["n_of_reviews"].fillna(0)
    loves = out["n_of_loves"].fillna(0)
    review_score = out["review_score"].fillna(0)
    price_per_ounce_clean = out["price_per_ounce"].fillna(0)

    # 1. Scor de popularitate 
    out["popularity_score"] = np.log1p(reviews) + np.log1p(loves)
    
    # 2. Scor engagement
    out["engagement_score"] = loves / (reviews + 1)
    
    # 3. Scor valoare (Calitate/Pret) - PROTEJAT la diviziune cu zero
    out["value_score"] = np.where(
        price_per_ounce_clean > 0, 
        review_score / price_per_ounce_clean, 
        0
    )
    
    # 4. Scor solid review_strength
    out["review_strength"] = review_score * np.log1p(reviews)
    
    # Curățare finală pentru ML
    return out.replace([np.inf, -np.inf], 0).fillna(0)