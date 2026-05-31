import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from Src.config import MODEL_FEATURES
from Src.feature_engineering import add_engineered_features


def prepare_similarity_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the dataset for the auxiliary similar-products module.
    Adds engineered features and drops rows with missing values in
    any of the MODEL_FEATURES columns.
    """
    out = df.copy()
    out = add_engineered_features(out)
    out = out.dropna(subset=MODEL_FEATURES).copy()
    return out


def compute_similarity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the pairwise cosine similarity matrix between products
    based on MODEL_FEATURES.
    Each cell (i, j) in the result is a similarity score in [0, 1]
    representing how similar product i is to product j.
    """
    features = df[MODEL_FEATURES].copy()

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    similarity = cosine_similarity(scaled_features)

    similarity_df = pd.DataFrame(
        similarity,
        index=df.index,
        columns=df.index
    )

    return similarity_df


def find_top_similar_products(
    df: pd.DataFrame,
    product_name: str,
    product_brand: str | None = None,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Return the top-N most similar products for a product identified by name
    (and optionally by brand).
    """
    if top_n <= 0:
        raise ValueError("top_n must be a positive integer.")

    if "name" not in df.columns:
        raise ValueError("Column 'name' is required for product identification.")

    if product_brand is not None and "brand" not in df.columns:
        raise ValueError("Column 'brand' is required when product_brand is provided.")

    prepared_df = prepare_similarity_dataframe(df)
    similarity_df = compute_similarity_matrix(prepared_df)

    if product_brand is not None:
        matches = prepared_df[
            (prepared_df["name"] == product_name) &
            (prepared_df["brand"] == product_brand)
        ]
        if matches.empty:
            raise ValueError(
                f"Product '{product_brand} - {product_name}' not found in the dataset."
            )
    else:
        matches = prepared_df[prepared_df["name"] == product_name]
        if matches.empty:
            raise ValueError(f"Product '{product_name}' not found in the dataset.")

    product_index = matches.index[0]
    similarity_scores = similarity_df.loc[product_index].drop(product_index)

    top_indices = similarity_scores.sort_values(ascending=False).head(top_n).index

    result_columns = [
        col for col in ["brand", "name", "price", "review_score", "price_per_ounce"]
        if col in prepared_df.columns
    ]

    result = prepared_df.loc[top_indices, result_columns].copy()
    result["similarity_score"] = similarity_scores.loc[top_indices].values

    return result.sort_values(by="similarity_score", ascending=False)