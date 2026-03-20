import numpy as np
import pandas as pd


def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adaugă log_reviews și log_loves pentru a reduce asimetria
    variabilelor n_of_reviews și n_of_loves.
    """
    required = ["n_of_reviews", "n_of_loves"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Lipsesc coloane necesare pentru add_log_features: {missing}")

    out = df.copy()

    negative_reviews = out["n_of_reviews"].dropna() < 0
    negative_loves = out["n_of_loves"].dropna() < 0

    if negative_reviews.any():
        raise ValueError(
            "Coloana 'n_of_reviews' conține valori negative, ceea ce este invalid pentru log1p."
        )

    if negative_loves.any():
        raise ValueError(
            "Coloana 'n_of_loves' conține valori negative, ceea ce este invalid pentru log1p."
        )

    out["log_reviews"] = np.log1p(out["n_of_reviews"])
    out["log_loves"] = np.log1p(out["n_of_loves"])
    return out


class ScoreScaler:
    """
    Memorează min/max pe TRAIN pentru a aplica aceeași normalizare și pe TEST.
    """

    def __init__(self):
        self.mins_ = {}
        self.maxs_ = {}

    def fit(self, df: pd.DataFrame, cols: list[str]):
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Lipsesc coloane necesare pentru ScoreScaler.fit: {missing}")

        for c in cols:
            series = df[c].dropna()

            if series.empty:
                raise ValueError(
                    f"Coloana '{c}' conține doar valori lipsă și nu poate fi folosită în ScoreScaler.fit."
                )

            self.mins_[c] = float(series.min())
            self.maxs_[c] = float(series.max())

        return self

    def transform_series(self, s: pd.Series, col: str) -> pd.Series:
        if col not in self.mins_ or col not in self.maxs_:
            raise ValueError(
                f"ScoreScaler nu este pregătit pentru coloana '{col}'. "
                f"Asigură-te că fit() a fost apelat corect."
            )

        mn = self.mins_[col]
        mx = self.maxs_[col]
        denom = (mx - mn) if (mx - mn) != 0 else 1.0
        x = (s - mn) / denom

        return x.clip(0, 1)


def compute_score_with_scaler(df: pd.DataFrame, scaler: ScoreScaler) -> pd.DataFrame:
    """
    Calculează ScorFinal (0-100) folosind normalizarea min-max
    învățată pe setul de train.

    Necesită coloanele:
    - review_score
    - log_reviews
    - log_loves
    - price_per_ounce
    """
    required = ["review_score", "log_reviews", "log_loves", "price_per_ounce"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Lipsesc coloane necesare pentru scoring: {missing}")

    out = df.copy()

    out["score_rating"] = scaler.transform_series(out["review_score"], "review_score")
    out["score_reviews"] = scaler.transform_series(out["log_reviews"], "log_reviews")
    out["score_loves"] = scaler.transform_series(out["log_loves"], "log_loves")
    out["score_price"] = 1 - scaler.transform_series(out["price_per_ounce"], "price_per_ounce")

    out["ScorFinal"] = 100 * (
        0.50 * out["score_rating"] +
        0.20 * out["score_reviews"] +
        0.20 * out["score_loves"] +
        0.10 * out["score_price"]
    )

    return out


def label_with_threshold(df_scored: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Aplică eticheta Merita pe baza unui prag deja calculat pe TRAIN.
    """
    out = df_scored.copy()
    out["Merita"] = (out["ScorFinal"] >= threshold).astype(int)
    return out