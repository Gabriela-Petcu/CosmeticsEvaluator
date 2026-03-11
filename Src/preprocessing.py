from __future__ import annotations
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, StandardScaler
from Src.config import LOG_FEATURE_COLUMNS, STANDARD_FEATURE_COLUMNS


def build_preprocessing_pipeline() -> ColumnTransformer:
    """
    pipeline de preprocessing: tratarea nan, transf variabilelor, scalarea lor
    """
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
        ("log_cols", log_pipe, LOG_FEATURE_COLUMNS),
        ("std_cols", std_pipe, STANDARD_FEATURE_COLUMNS)
    ])