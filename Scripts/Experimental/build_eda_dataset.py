from Src.feature_engineering import add_engineered_features
from Src.inference import load_bundle
from Src.io import load_skincare_dv
from Src.scoring import add_log_features, compute_score_with_scaler
from Src.config import PROCESSED_DIR


def main():
    # creează datasetul pentru EDA
    df = load_skincare_dv()
    df = add_engineered_features(df)
    df = add_log_features(df)

    bundle = load_bundle()
    scaler = bundle["score_scaler"]

    df = compute_score_with_scaler(df, scaler)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "skincare_eda_scored.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved (EDA only): {out_path}")


if __name__ == "__main__":
    main()