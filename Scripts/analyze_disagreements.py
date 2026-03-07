import pandas as pd
import joblib
from pathlib import Path

from Src.io import load_skincare_dv
from Src.scoring import (
    add_log_features,
    ScoreScaler,
    compute_score_with_scaler,
    label_with_threshold
)

MODEL_PATH = "Models/bundle_v1.joblib"


def main():

    print("Loading dataset...")
    df = load_skincare_dv()

    # -------------------------
    # BASELINE
    # -------------------------

    df = add_log_features(df)

    score_cols = ["review_score", "log_reviews", "log_loves", "price_per_ounce"]
    scaler = ScoreScaler().fit(df, cols=score_cols)

    df_scored = compute_score_with_scaler(df, scaler)

    threshold = float(df_scored["ScorFinal"].quantile(0.75))
    df_labeled = label_with_threshold(df_scored, threshold)

    # -------------------------
    # LOAD MODEL
    # -------------------------

    print("Loading trained model...")
    bundle = joblib.load(MODEL_PATH)

    pipeline = bundle["full_system"]

    features = ["n_of_reviews", "n_of_loves", "review_score", "price_per_ounce"]

    # -------------------------
    # ML PREDICTIONS
    # -------------------------

    df_labeled["MeritaML"] = pipeline.predict(df_labeled[features])

    df_labeled["ProbabilitateML"] = pipeline.predict_proba(df_labeled[features])[:, 1]

    # -------------------------
    # DISAGREEMENTS
    # -------------------------

    df_labeled["Disagreement"] = df_labeled["Merita"] != df_labeled["MeritaML"]

    total = len(df_labeled)

    disagreements = df_labeled["Disagreement"].sum()

    agreement = total - disagreements

    disagreement_rate = disagreements / total

    print("\n=== BASELINE vs ML COMPARISON ===")

    print(f"Total products: {total}")

    print(f"Agreement: {agreement}")

    print(f"Disagreements: {disagreements}")

    print(f"Disagreement rate: {disagreement_rate:.4f}")

    # -------------------------
    # TYPE OF DISAGREEMENTS
    # -------------------------

    baseline_0_ml_1 = df_labeled[
        (df_labeled["Merita"] == 0) &
        (df_labeled["MeritaML"] == 1)
    ]

    baseline_1_ml_0 = df_labeled[
        (df_labeled["Merita"] == 1) &
        (df_labeled["MeritaML"] == 0)
    ]

    print("\nType of disagreements:")

    print(f"Baseline=0 & ML=1: {len(baseline_0_ml_1)}")

    print(f"Baseline=1 & ML=0: {len(baseline_1_ml_0)}")

    # -------------------------
    # SAVE RESULTS
    # -------------------------

    output_dir = Path("Data/Processed")

    output_dir.mkdir(parents=True, exist_ok=True)

    summary = pd.DataFrame({
        "metric": [
            "total_products",
            "agreement",
            "disagreements",
            "disagreement_rate",
            "baseline_0_ml_1",
            "baseline_1_ml_0"
        ],
        "value": [
            total,
            agreement,
            disagreements,
            disagreement_rate,
            len(baseline_0_ml_1),
            len(baseline_1_ml_0)
        ]
    })

    summary_path = output_dir / "disagreements_summary.csv"

    summary.to_csv(summary_path, index=False)

    examples = df_labeled[df_labeled["Disagreement"]].copy()

    examples_path = output_dir / "disagreements_examples.csv"

    examples.to_csv(examples_path, index=False)

    print("\nSaved files:")

    print(summary_path)

    print(examples_path)


if __name__ == "__main__":
    main()