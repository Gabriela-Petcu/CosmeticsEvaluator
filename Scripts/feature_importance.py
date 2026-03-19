import joblib
import pandas as pd

from Src.config import MODEL_PATH, MODEL_FEATURES


def load_model():
    bundle = joblib.load(MODEL_PATH)
    pipeline = bundle["full_system"]

    model = pipeline.named_steps["classifier"]
    preprocessor = pipeline.named_steps["preprocessor"]

    return model, preprocessor


def get_feature_names(preprocessor):
    """
    Extrage numele feature-urilor după preprocessing
    """
    feature_names = []

    for name, transformer, columns in preprocessor.transformers_:
        if name == "log_cols":
            feature_names.extend(columns)
        elif name == "std_cols":
            feature_names.extend(columns)

    return feature_names


def analyze_feature_importance():
    model, preprocessor = load_model()

    coefficients = model.coef_[0]
    feature_names = get_feature_names(preprocessor)

    df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
        "abs_importance": abs(coefficients)
    })

    df = df.sort_values(by="abs_importance", ascending=False)

    print("\n=== FEATURE IMPORTANCE (LOGISTIC REGRESSION) ===")
    print(df.to_string(index=False))

    print("\n=== TOP POSITIVE FEATURES (favorizează 'Merită') ===")
    print(df.sort_values(by="coefficient", ascending=False).head(5).to_string(index=False))

    print("\n=== TOP NEGATIVE FEATURES (defavorizează 'Merită') ===")
    print(df.sort_values(by="coefficient", ascending=True).head(5).to_string(index=False))


def main():
    analyze_feature_importance()


if __name__ == "__main__":
    main()