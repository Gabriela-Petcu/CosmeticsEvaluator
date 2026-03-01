import pandas as pd
from .config import RAW_SKINCARE_DV

def load_skincare_dv(path=RAW_SKINCARE_DV) -> pd.DataFrame:
    """
    Încarcă dataset-ul skincare_dv.csv într-un DataFrame.
    """
    return pd.read_csv(path)
