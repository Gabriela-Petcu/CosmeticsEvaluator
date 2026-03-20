import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from Src.config import MODEL_FEATURES
from Src.feature_engineering import add_engineered_features


def prepare_similarity_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Pregătește datasetul pentru un modul auxiliar de produse similare.
    Nu face parte din verdictul final al aplicației, ci este folosit doar
    pentru identificarea produselor apropiate ca profil numeric.
    """
    out = df.copy()
    out = add_engineered_features(out)
    out = out.dropna(subset=MODEL_FEATURES).copy()
    return out


def compute_similarity_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculează similaritatea cosinus între produse pe baza MODEL_FEATURES.
    Această componentă este auxiliară și nu influențează verdictul final
    de recomandare al aplicației.
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
    Returnează primele top_n produse similare pentru un produs identificat prin nume.
    Dacă product_brand este oferit, selecția se face după nume + brand.

    Dacă există mai multe produse cu același nume și product_brand nu este furnizat,
    este utilizată prima potrivire găsită în dataset.
    """
    if top_n <= 0:
        raise ValueError("top_n trebuie să fie un număr pozitiv.")

    if "name" not in df.columns:
        raise ValueError("Coloana 'name' este necesară pentru identificarea produsului.")

    if product_brand is not None and "brand" not in df.columns:
        raise ValueError("Coloana 'brand' este necesară când product_brand este furnizat.")

    prepared_df = prepare_similarity_dataframe(df)
    similarity_df = compute_similarity_matrix(prepared_df)

    if product_brand is not None:
        matches = prepared_df[
            (prepared_df["name"] == product_name) &
            (prepared_df["brand"] == product_brand)
        ]
        if matches.empty:
            raise ValueError(
                f"Produsul '{product_brand} - {product_name}' nu a fost găsit în dataset."
            )
    else:
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