import joblib
from pathlib import Path
import pandas as pd

from Src.io import load_skincare_dv
from Src.scoring import add_log_features, compute_score_with_scaler, label_with_threshold

bundle = joblib.load("Models/bundle_v1.joblib")
full_system = bundle["full_system"]
threshold = float(bundle["threshold"])
scaler = bundle["score_scaler"]

df = load_skincare_dv()
df = add_log_features(df)

df_scored = compute_score_with_scaler(df, scaler)
df_scored = label_with_threshold(df_scored, threshold)

features = ["n_of_reviews", "n_of_loves", "review_score", "price"]
df_scored["MeritaML"] = full_system.predict(df_scored[features])
df_scored["ProbabilitateML"] = full_system.predict_proba(df_scored[features])[:, 1]

name_col = None
for c in ["name", "product_name", "product", "title"]:
    if c in df_scored.columns:
        name_col = c
        break

output_cols = []
if name_col:
    output_cols.append(name_col)

output_cols += ["ScorFinal", "Merita", "MeritaML", "ProbabilitateML"]

Path("Data/Processed").mkdir(parents=True, exist_ok=True)
out_path = Path("Data/Processed/comparison_results.csv")
df_scored[output_cols].to_csv(out_path, index=False)

print(f"✅ Rezultate comparate salvate în: {out_path}")