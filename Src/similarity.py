import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from Src.config import MODEL_FEATURES
from Src.feature_engineering import add_engineered_features


def prepare_similarity_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pregătește datasetul pentru calculul similarității între produse.
    Adaugă feature-urile construite și elimină produsele fără valorile necesare.
    """
    df = df.copy()
    df = add_engineered_features(df)
    df = df.dropna(subset=MODEL_FEATURES).copy()
    return df


def compute_similarity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculează matricea de similaritate cosinus între produse,
    folosind MODEL_FEATURES.
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


def get_top_similar_products(
    df: pd.DataFrame,
    product_name: str,
    product_brand: str | None = None,
    top_n: int = 5
) -> pd.DataFrame:
    """
    Returnează primele top_n produse similare pentru un produs identificat prin nume.
    """
    prepared_df = prepare_similarity_dataframe(df)
    similarity_df = compute_similarity_matrix(prepared_df)

    matches = prepared_df[prepared_df["name"] == product_name]
    if matches.empty:
        raise ValueError(f"Produsul '{product_name}' nu a fost găsit în dataset.")

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