import pandas as pd

from Src.io import load_skincare_dv
from Src.scoring import add_log_features, compute_score_with_scaler
from Src.feature_engineering import add_engineered_features
from Src.config import SCORE_COLUMNS
from Src.scoring import ScoreScaler


def prepare_dataset():
    df = load_skincare_dv()

    df = add_engineered_features(df)
    df = add_log_features(df)

    scaler = ScoreScaler().fit(df, cols=SCORE_COLUMNS)
    df = compute_score_with_scaler(df, scaler)

    df = df.dropna(subset=["ScorFinal", "price_per_ounce"]).copy()

    return df


def analyze_price_vs_quality(df: pd.DataFrame):
    print("\n=== BASIC STATS ===")
    print(df[["price_per_ounce", "ScorFinal"]].describe())

    correlation = df["price_per_ounce"].corr(df["ScorFinal"])
    print(f"\nCorrelation (price vs quality): {correlation:.4f}")

    # produse cu cel mai bun raport calitate / pret
    df["value_ratio"] = df["ScorFinal"] / df["price_per_ounce"]

    top_value = df.sort_values(by="value_ratio", ascending=False).head(10)

    print("\n=== TOP 10 VALUE PRODUCTS ===")
    print(
        top_value[[
            "brand",
            "name",
            "price",
            "price_per_ounce",
            "ScorFinal",
            "value_ratio"
        ]].to_string(index=False)
    )

    # produse scumpe dar slabe
    worst_value = df.sort_values(by="value_ratio", ascending=True).head(10)

    print("\n=== WORST VALUE PRODUCTS ===")
    print(
        worst_value[[
            "brand",
            "name",
            "price",
            "price_per_ounce",
            "ScorFinal",
            "value_ratio"
        ]].to_string(index=False)
    )

    return df


def main():
    df = prepare_dataset()
    df = analyze_price_vs_quality(df)


if __name__ == "__main__":
    main()