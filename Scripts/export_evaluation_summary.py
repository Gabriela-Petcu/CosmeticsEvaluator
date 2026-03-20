import pandas as pd

from Src.config import PROCESSED_DIR


def read_summary_metric(df: pd.DataFrame, metric_name: str) -> float:
    row = df[df["metric"] == metric_name]
    if row.empty:
        raise ValueError(f"Nu am găsit metrica '{metric_name}' în fișierul summary.")
    return float(row.iloc[0]["mean"] if "mean" in row.columns else row.iloc[0]["value"])


def main():
    cv_summary_path = PROCESSED_DIR / "cross_validation_summary.csv"
    disagreements_path = PROCESSED_DIR / "disagreements_summary.csv"
    importance_path = PROCESSED_DIR / "feature_importance.csv"
    shap_examples_path = PROCESSED_DIR / "shap_examples.csv"
    recommendation_examples_path = PROCESSED_DIR / "user_recommendation_examples.csv"

    cv_summary = pd.read_csv(cv_summary_path)
    disagreements = pd.read_csv(disagreements_path)
    importance = pd.read_csv(importance_path)
    shap_examples = pd.read_csv(shap_examples_path)
    recommendation_examples = pd.read_csv(recommendation_examples_path)

    accuracy = read_summary_metric(cv_summary, "accuracy")
    precision = read_summary_metric(cv_summary, "precision")
    recall = read_summary_metric(cv_summary, "recall")
    f1 = read_summary_metric(cv_summary, "f1")

    total_products = int(read_summary_metric(disagreements, "total_products"))
    agreement = int(read_summary_metric(disagreements, "agreement"))
    disagreement_count = int(read_summary_metric(disagreements, "disagreements"))
    disagreement_rate = read_summary_metric(disagreements, "disagreement_rate")

    top_features = importance.head(4)

    lines = [
        "# Evaluation Summary",
        "",
        "## 1. Cross-validation results",
        f"- Accuracy (mean): {accuracy:.4f}",
        f"- Precision (mean): {precision:.4f}",
        f"- Recall (mean): {recall:.4f}",
        f"- F1-score (mean): {f1:.4f}",
        "",
        "## 2. Baseline vs ML comparison",
        f"- Total products analyzed: {total_products}",
        f"- Agreement: {agreement}",
        f"- Disagreements: {disagreement_count}",
        f"- Disagreement rate: {disagreement_rate:.4f}",
        "",
        "## 3. Logistic Regression feature importance",
    ]

    for _, row in top_features.iterrows():
        lines.append(
            f"- {row['feature']}: coef={row['coefficient']:.6f}, abs_importance={row['abs_importance']:.6f}"
        )

    lines += [
        "",
        "## 4. SHAP representative examples",
    ]

    for _, row in shap_examples.iterrows():
        lines.append(
            f"- {row['brand']} - {row['name']} | "
            f"Merita={row['Merita']} | MeritaML={row['MeritaML']} | "
            f"ProbabilitateML={row['ProbabilitateML']:.4f}"
        )

    lines += [
        "",
        "## 5. User recommendation scenarios",
    ]

    for _, row in recommendation_examples.iterrows():
        lines.append(
            f"- Profil: ({row['skin_type']}, {row['main_concern']}, {row['budget_level']}) | "
            f"Produs: {row['brand']} - {row['name']} | "
            f"VerdictFinal: {row['VerdictFinal']}"
        )

    output_path = PROCESSED_DIR / "evaluation_summary.md"
    output_path.write_text("\n".join(lines), encoding="utf-8")

    print(f" Evaluation summary saved to: {output_path}")


if __name__ == "__main__":
    main()