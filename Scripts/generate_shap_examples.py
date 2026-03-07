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
from Src.explainability import explain_product, explanation_to_dict


MODEL_PATH = "Models/bundle_v1.joblib"


def build_full_analysis_df() -> pd.DataFrame:
    df = load_skincare_dv()

    df = add_log_features(df)

    score_cols = ["review_score", "log_reviews", "log_loves", "price_per_ounce"]
    scaler = ScoreScaler().fit(df, cols=score_cols)

    df_scored = compute_score_with_scaler(df, scaler)
    threshold = float(df_scored["ScorFinal"].quantile(0.75))
    df_labeled = label_with_threshold(df_scored, threshold)

    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["full_system"]

    features = ["n_of_reviews", "n_of_loves", "review_score", "price_per_ounce"]

    df_labeled["MeritaML"] = pipeline.predict(df_labeled[features])
    df_labeled["ProbabilitateML"] = pipeline.predict_proba(df_labeled[features])[:, 1]
    df_labeled["Disagreement"] = df_labeled["Merita"] != df_labeled["MeritaML"]
    df_labeled["DistanceToThreshold"] = (df_labeled["ScorFinal"] - threshold).abs()

    return df_labeled


def pick_examples(df: pd.DataFrame) -> pd.DataFrame:
    # 1. Produs clar bun: baseline=1, ML=1, probabilitate mare
    good_df = df[(df["Merita"] == 1) & (df["MeritaML"] == 1)].copy()
    good_df = good_df.sort_values(
        by=["ProbabilitateML", "ScorFinal"],
        ascending=[False, False]
    )
    good_example = good_df.head(1)

    # 2. Produs clar slab: baseline=0, ML=0, probabilitate mică
    weak_df = df[(df["Merita"] == 0) & (df["MeritaML"] == 0)].copy()
    weak_df = weak_df.sort_values(
        by=["ProbabilitateML", "ScorFinal"],
        ascending=[True, True]
    )
    weak_example = weak_df.head(1)

    # 3. Produs interesant: disagreement sau aproape de threshold
    disagreement_df = df[df["Disagreement"]].copy()

    if not disagreement_df.empty:
        interesting_df = disagreement_df.sort_values(
            by=["ProbabilitateML", "DistanceToThreshold"],
            ascending=[False, True]
        )
        interesting_example = interesting_df.head(1)
    else:
        borderline_df = df.copy().sort_values(by="DistanceToThreshold", ascending=True)
        interesting_example = borderline_df.head(1)

    examples = pd.concat([good_example, weak_example, interesting_example], axis=0)
    examples = examples.drop_duplicates(subset=["brand", "name"])
    return examples


def build_explanations_table(examples_df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    feature_cols = ["n_of_reviews", "n_of_loves", "review_score", "price_per_ounce"]

    for _, row in examples_df.iterrows():
        product_input = row[feature_cols].to_dict()
        explanation = explain_product(product_input)
        explanation_dict = explanation_to_dict(explanation)

        top_factors = explanation_dict["TopFactori"]

        rows.append({
            "brand": row.get("brand", ""),
            "name": row.get("name", ""),
            "price": row.get("price", None),
            "price_per_ounce": row.get("price_per_ounce", None),
            "n_of_reviews": row.get("n_of_reviews", None),
            "n_of_loves": row.get("n_of_loves", None),
            "review_score": row.get("review_score", None),
            "ScorFinal": explanation_dict["ScorFinal"],
            "Merita": explanation_dict["Merita"],
            "MeritaML": explanation_dict["MeritaML"],
            "ProbabilitateML": explanation_dict["ProbabilitateML"],
            "TopFactor1": top_factors[0]["feature"] if len(top_factors) > 0 else None,
            "TopFactor1Direction": top_factors[0]["direction"] if len(top_factors) > 0 else None,
            "TopFactor1SHAP": top_factors[0]["shap_value"] if len(top_factors) > 0 else None,
            "TopFactor2": top_factors[1]["feature"] if len(top_factors) > 1 else None,
            "TopFactor2Direction": top_factors[1]["direction"] if len(top_factors) > 1 else None,
            "TopFactor2SHAP": top_factors[1]["shap_value"] if len(top_factors) > 1 else None,
            "TopFactor3": top_factors[2]["feature"] if len(top_factors) > 2 else None,
            "TopFactor3Direction": top_factors[2]["direction"] if len(top_factors) > 2 else None,
            "TopFactor3SHAP": top_factors[2]["shap_value"] if len(top_factors) > 2 else None,
        })

    return pd.DataFrame(rows)


def main():
    print("Loading and analyzing full dataset...")
    full_df = build_full_analysis_df()

    print("Selecting representative examples...")
    examples_df = pick_examples(full_df)

    print("Generating SHAP explanations...")
    explanations_df = build_explanations_table(examples_df)

    output_dir = Path("Data") / "Processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "shap_examples.csv"
    explanations_df.to_csv(output_path, index=False)

    print("\n=== SHAP EXAMPLES ===")
    print(explanations_df.to_string(index=False))
    print(f"\n✅ Saved to: {output_path}")


if __name__ == "__main__":
    main()