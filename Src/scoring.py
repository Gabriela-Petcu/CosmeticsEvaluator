import numpy as np
import pandas as pd


def add_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add log_reviews and log_loves columns to the DataFrame by applying
    a log1p transformation to reduce the skewness of n_of_reviews
    and n_of_loves.
    Parameters
    df : pd.DataFrame
        Input DataFrame containing non-negative columns
        'n_of_reviews' and 'n_of_loves'.
 
    Returns
    pd.DataFrame
        Copy of the input DataFrame with two new columns:
        - 'log_reviews' : log1p of 'n_of_reviews'
        - 'log_loves'   : log1p of 'n_of_loves'
    """
    required = ["n_of_reviews", "n_of_loves"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required for add_log_features: {missing}")

    out = df.copy()

    negative_reviews = out["n_of_reviews"].dropna() < 0
    negative_loves = out["n_of_loves"].dropna() < 0

    if negative_reviews.any():
        raise ValueError(
            "Column 'n_of_reviews' contains negative values, which is invalid for log1p."
        )
    if negative_loves.any():
        raise ValueError(
            "Column 'n_of_loves' contains negative values, which are invalid for log1p."
        )
    out["log_reviews"] = np.log1p(out["n_of_reviews"])
    out["log_loves"] = np.log1p(out["n_of_loves"])
    return out


class ScoreScaler:
    """
    Min-max scaler that memorizes the minimum and maximum values from the train set and applies the same normalization to test set. 
    Attributes
    mins_ : dict
        Per-column minimum values, populated after fit().
    maxs_ : dict
        Per-column maximum values, populated after fit().
    _is_fitted : bool
        Internal flag indicating whether fit() has been called.
    """

    def __init__(self):
        self.mins_ = {}
        self.maxs_ = {}
        self._is_fitted = False

    def fit(self, df: pd.DataFrame, cols: list[str]):
        """
        Compute and store the min and max for each column in cols using the train set.
        Parameters
        df : pd.DataFrame
            Training DataFrame from which statistics are computed.
        cols : list of str
            List of column names to be scaled.
        Returns
        self : ScoreScaler
        """
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns required for ScoreScaler.fit: {missing}")

        for c in cols:
            series = df[c].dropna()

            if series.empty:
                raise ValueError(
                    f"Column '{c}' contains only missing values and cannot be used in ScoreScaler.fit."
                )
            self.mins_[c] = float(series.min())
            self.maxs_[c] = float(series.max())
        self._is_fitted = True
        return self

    def _check_is_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError(
                "ScoreScaler has not been fitted. Call fit() before transform_series()."
            )
        
    def transform_series(self, s: pd.Series, col: str) -> pd.Series:
        """
        Apply the min-max normalization learned on train to a new series. 
        Parameters
        s : pd.Series
            The series to be transformed.
        col : str
            Name of the corresponding column (must have been seen in fit()).
        Returns
        pd.Series
            The transformed series.
        """
        self._check_is_fitted()

        if col not in self.mins_ or col not in self.maxs_:
            raise ValueError(
                f"Column '{col}' was not seen in fit(). "
                f"Available columns: {list(self.mins_.keys())}"
            )
        mn = self.mins_[col]
        mx = self.maxs_[col]
        denom = (mx - mn) if (mx - mn) != 0 else 1.0
        x = (s - mn) / denom

        return x.clip(0, 1)


def compute_score_with_scaler(df: pd.DataFrame, scaler: ScoreScaler) -> pd.DataFrame:
    """
    Calculate Final Score (0-100) using min-max normalization
    learned on the training set.
    Formula:
        FinalScore = 100 × (0.50 × score_rating
                           + 0.20 × score_reviews
                           + 0.20 × score_loves
                           + 0.10 × score_price)
    Parametres:
    df : pd.DataFrame
        Input DataFrame containing the columns: 'review_score', 'log_reviews', 'log_loves', 'price_per_ounce'.
    scaler : ScoreScaler
        A fitted ScoreScaler instance (fit() already called on train data).
    Returns
    pd.DataFrame
        Copy of the input DataFrame with intermediate columns added
        ('score_rating', 'score_reviews', 'score_loves', 'score_price')
        and the final 'FinalScore' column.
    """
    if not isinstance(scaler, ScoreScaler):
        raise TypeError(f"scaler must be an instance of ScoreScaler, not {type(scaler).__name__}")
    required = ["review_score", "log_reviews", "log_loves", "price_per_ounce"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns required for scoring: {missing}")

    out = df.copy()

    out["score_rating"] = scaler.transform_series(out["review_score"], "review_score")
    out["score_reviews"] = scaler.transform_series(out["log_reviews"], "log_reviews")
    out["score_loves"] = scaler.transform_series(out["log_loves"], "log_loves")
    out["score_price"] = 1 - scaler.transform_series(out["price_per_ounce"], "price_per_ounce")

    out["FinalScore"] = 100 * (
        0.50 * out["score_rating"] +
        0.20 * out["score_reviews"] +
        0.20 * out["score_loves"] +
        0.10 * out["score_price"]
    )

    return out


def label_with_threshold(df_scored: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Assign a binary IsRecommended label to each product by comparing
    FinalScore against a threshold pre-computed on the training set.
    Parameters
    df_scored : pd.DataFrame
        Input DataFrame containing the 'FinalScore' column.
    threshold : float
        The threshold value for labeling. Products with FinalScore >= threshold
        will be labeled as 1 (recommended), and those below will be labeled as 0 (not recommended).
    Returns
    Copy of the input DataFrame with 'IsRecommended' added
    """
    out = df_scored.copy()
    out["IsRecommended"] = (out["FinalScore"] >= threshold).astype(int)
    return out