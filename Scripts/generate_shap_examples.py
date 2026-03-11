import pandas as pd

from Src.explainability import explain_product, explanation_to_dict
from Src.config import PROCESSED_DIR, MODEL_FEATURES
from Src.inference import build_full_analysis_df, load_bundle


def pick_examples(df: pd.DataFrame) -> pd.DataFrame:
    good_df = df[(df["Merita"] == 1) & (df["MeritaML"] == 1)].copy()
    good_df = good_df.sort_values(
        by=["ProbabilitateML", "ScorFinal"],
        ascending=[False, False]
    )
    good_example = good_df.head(1)

    weak_df = df[(df["Merita"] == 0) & (df["MeritaML"] == 0)].copy()
    weak_df = weak_df.sort_values(
        by=["ProbabilitateML", "ScorFinal"],
        ascending=[True, True]
    )
    weak_example = weak_df.head(1)

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

    for _, row in examples_df.iterrows():
        product_input = row[MODEL_FEATURES].to_dict()
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

    bundle = load_bundle()
    threshold = float(bundle["threshold"])

    full_df["Disagreement"] = full_df["Merita"] != full_df["MeritaML"]
    full_df["DistanceToThreshold"] = (full_df["ScorFinal"] - threshold).abs()

    print("Selecting representative examples...")
    examples_df = pick_examples(full_df)

    print("Generating SHAP explanations...")
    explanations_df = build_explanations_table(examples_df)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / "shap_examples.csv"
    explanations_df.to_csv(output_path, index=False)

    print("\n=== SHAP EXAMPLES ===")
    print(explanations_df.to_string(index=False))
    print(f"\n Saved to: {output_path}")


if __name__ == "__main__":
    main()