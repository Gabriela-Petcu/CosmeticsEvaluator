from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
import numpy as np

def build_preprocessing_pipeline(log_cols, std_cols):
    log_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("log", FunctionTransformer(np.log1p, validate=False)),
        ("scaler", MinMaxScaler())
    ])

    std_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    return ColumnTransformer([
        ("log_cols", log_pipe, log_cols),
        ("std_cols", std_pipe, std_cols)
    ])