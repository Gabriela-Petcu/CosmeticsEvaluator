from Src.io import load_skincare_dv
from Src.config import PROCESSED_DIR
from Src.scoring import add_log_features, ScoreScaler, compute_score_with_scaler

def main():
    df = load_skincare_dv()
    df = add_log_features(df)

    score_cols = ["review_score", "log_reviews", "log_loves", "price_per_ounce"]
    scaler = ScoreScaler().fit(df, cols=score_cols)

    df_scored = compute_score_with_scaler(df, scaler)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "skincare_eda_scored.csv"
    df_scored.to_csv(out_path, index=False)

    print(f"Saved (EDA only): {out_path}")

if __name__ == "__main__":
    main()