import pandas as pd

from Src.io import load_skincare_dv
from Src.config import MODEL_FEATURES
from Src.feature_engineering import add_engineered_features
from Src.explainability import explain_product, print_explanation


def main():
    df = load_skincare_dv()
    df = add_engineered_features(df)

    df_valid = df.dropna(subset=MODEL_FEATURES).copy()

    if df_valid.empty:
        raise ValueError("Nu există produse valide pentru explainability după feature engineering.")

    product = df_valid.iloc[0]

    print("=== DEMO EXPLAINABILITY ===")
    print(f"Produs: {product.get('brand', '')} - {product.get('name', '')}")
    print(f"Price: {product.get('price', 'N/A')}")
    print(f"Review score: {product.get('review_score', 'N/A')}")
    print(f"Reviews: {product.get('n_of_reviews', 'N/A')}")
    print(f"Loves: {product.get('n_of_loves', 'N/A')}")
    print()

    result = explain_product(product.to_dict())
    print_explanation(result)


if __name__ == "__main__":
    main()