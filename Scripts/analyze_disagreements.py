import pandas as pd

from Src.inference import build_full_analysis_df
from Src.config import PROCESSED_DIR


def main():
    print("Loading full analysis dataset...")
    df_labeled = build_full_analysis_df()

    df_labeled["Disagreement"] = df_labeled["Merita"] != df_labeled["MeritaML"]

    total = len(df_labeled)
    disagreements = int(df_labeled["Disagreement"].sum())
    agreement = total - disagreements
    disagreement_rate = disagreements / total

    print("\n=== BASELINE vs ML COMPARISON ===")
    print(f"Total products: {total}")
    print(f"Agreement: {agreement}")
    print(f"Disagreements: {disagreements}")
    print(f"Disagreement rate: {disagreement_rate:.4f}")

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

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

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

    summary_path = PROCESSED_DIR / "disagreements_summary.csv"
    summary.to_csv(summary_path, index=False)

    examples = df_labeled[df_labeled["Disagreement"]].copy()
    examples_path = PROCESSED_DIR / "disagreements_examples.csv"
    examples.to_csv(examples_path, index=False)

    print("\nSaved files:")
    print(summary_path)
    print(examples_path)


if __name__ == "__main__":
    main()