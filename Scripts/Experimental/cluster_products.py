import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from Src.io import load_skincare_dv
from Src.feature_engineering import add_engineered_features
from Src.scoring import add_log_features, compute_score_with_scaler, ScoreScaler
from Src.config import SCORE_COLUMNS


def prepare_dataset():
    df = load_skincare_dv()

    df = add_engineered_features(df)
    df = add_log_features(df)

    scaler = ScoreScaler().fit(df, cols=SCORE_COLUMNS)
    df = compute_score_with_scaler(df, scaler)

    df = df.dropna(subset=[
        "price_per_ounce",
        "ScorFinal",
        "n_of_reviews",
        "n_of_loves"
    ]).copy()

    return df


def run_kmeans(df: pd.DataFrame, n_clusters: int = 3):
    features = df[[
        "price_per_ounce",
        "ScorFinal",
        "n_of_reviews",
        "n_of_loves"
    ]]

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X)

    return df


def analyze_clusters(df: pd.DataFrame):
    print("\n=== CLUSTER SUMMARY ===")

    summary = df.groupby("cluster")[[
        "price_per_ounce",
        "ScorFinal",
        "n_of_reviews",
        "n_of_loves"
    ]].mean()

    print(summary)

    print("\n=== SAMPLE PRODUCTS PER CLUSTER ===")

    for cluster_id in sorted(df["cluster"].unique()):
        print(f"\n--- Cluster {cluster_id} ---")

        sample = df[df["cluster"] == cluster_id].head(5)

        print(sample[[
            "brand",
            "name",
            "price_per_ounce",
            "ScorFinal"
        ]].to_string(index=False))


def main():
    df = prepare_dataset()
    df = run_kmeans(df, n_clusters=3)
    analyze_clusters(df)


if __name__ == "__main__":
    main()