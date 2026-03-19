from __future__ import annotations
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler
from Src.config import LOG_FEATURE_COLUMNS, STANDARD_FEATURE_COLUMNS


def build_preprocessing_pipeline(
    log_feature_columns=None,
    standard_feature_columns=None
) -> ColumnTransformer:
    """
    Pipeline de preprocessing: tratarea valorilor lipsă,
    transformarea variabilelor și scalarea lor.
    Dacă nu se transmit liste de coloane, se folosesc cele din config.py.
    """
    if log_feature_columns is None:
        log_feature_columns = LOG_FEATURE_COLUMNS

    if standard_feature_columns is None:
        standard_feature_columns = STANDARD_FEATURE_COLUMNS

    log_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, validate=False, feature_names_out="one-to-one")),
        ("scaler", MinMaxScaler())
    ])

    std_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    return ColumnTransformer([
        ("log_cols", log_pipe, log_feature_columns),
        ("std_cols", std_pipe, standard_feature_columns)
    ])