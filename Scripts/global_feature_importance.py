from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt


MODEL_PATH = Path("Models") / "bundle_v1.joblib"
OUTPUT_DIR = Path("Data") / "Processed"


def clean_feature_name(feature_name: str) -> str:
    """
    Elimină prefixele generate de ColumnTransformer.
    Exemplu:
    log_cols__n_of_reviews -> n_of_reviews
    std_cols__price_per_ounce -> price_per_ounce
    """
    if "__" in feature_name:
        return feature_name.split("__", 1)[1]
    return feature_name


def main():
    # =========================
    # 1. Load trained bundle
    # =========================
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Nu există modelul la calea: {MODEL_PATH}. "
            "Rulează mai întâi python -m Scripts.train_model"
        )

    bundle = joblib.load(MODEL_PATH)
    full_system = bundle["full_system"]

    preprocessor = full_system.named_steps["preprocessor"]
    classifier = full_system.named_steps["classifier"]

    # =========================
    # 2. Get feature names + importances
    # =========================
    raw_feature_names = list(preprocessor.get_feature_names_out())
    clean_names = [clean_feature_name(name) for name in raw_feature_names]

    importances = classifier.feature_importances_

    importance_df = pd.DataFrame({
        "feature": clean_names,
        "importance": importances
    })

    importance_df = importance_df.sort_values(
        by="importance",
        ascending=False
    ).reset_index(drop=True)

    # =========================
    # 3. Save CSV
    # =========================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT_DIR / "global_feature_importance.csv"
    importance_df.to_csv(csv_path, index=False)

    # =========================
    # 4. Plot chart
    # =========================
    plt.figure(figsize=(8, 5))
    plt.bar(importance_df["feature"], importance_df["importance"])
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title("Global Feature Importance - RandomForest")
    plt.xticks(rotation=20)
    plt.tight_layout()

    png_path = OUTPUT_DIR / "global_feature_importance.png"
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()

    # =========================
    # 5. Print results
    # =========================
    print("=== GLOBAL FEATURE IMPORTANCE ===")
    print(importance_df.to_string(index=False))

    print(f"\n✅ CSV saved to: {csv_path}")
    print(f"✅ Chart saved to: {png_path}")


if __name__ == "__main__":
    main()