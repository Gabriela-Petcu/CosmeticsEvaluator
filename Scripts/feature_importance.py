import joblib
import pandas as pd

from Src.config import MODEL_PATH, PROCESSED_DIR


def load_model():
    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["full_system"]
    model = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]
    return model, preprocessor


def clean_feature_name(feature_name: str) -> str:
    """
    Strip ColumnTransformer prefixes from a feature name.
    Example: 'log_cols__n_of_reviews' -> 'n_of_reviews'
    """
    if "__" in feature_name:
        return feature_name.split("__", 1)[1]
    return feature_name


def analyze_feature_importance():
    """
    Extract and display logistic regression coefficients as feature importance.
    """
    model, preprocessor = load_model()

    coefficients = model.coef_[0]
    feature_names = [
        clean_feature_name(name)
        for name in preprocessor.get_feature_names_out()
    ]

    df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "abs_importance": abs(coefficients)
    })

    df = df.sort_values(by="abs_importance", ascending=False)

    positive_df = df[df["coefficient"] > 0].sort_values(by="coefficient", ascending=False)
    negative_df = df[df["coefficient"] < 0].sort_values(by="coefficient", ascending=True)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "feature_importance.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to: {out_path}")

    print("\n COEFFICIENT ANALYSIS (LOGISTIC REGRESSION) ")
    print(df.to_string(index=False))

    print("\n TOP POSITIVE COEFFICIENTS  ")
    print(positive_df.head(5).to_string(index=False))

    print("\nNEGATIVE COEFFICIENTS")
    if negative_df.empty:
        print("No negative coefficients found in the model.")
    else:
        print(negative_df.to_string(index=False))


def main():
    analyze_feature_importance()


if __name__ == "__main__":
    main()