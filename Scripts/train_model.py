import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from Src.config import PROJECT_ROOT, SCORE_COLUMNS, MODEL_FEATURES
from Src.io import load_skincare_dv
from Src.preprocessing import build_preprocessing_pipeline
from Src.scoring import add_log_features, ScoreScaler, compute_score_with_scaler, label_with_threshold
from Src.feature_engineering import add_engineered_features

def main():
    df = load_skincare_dv()
    
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42
    )

    train_df = add_engineered_features(train_df)
    test_df = add_engineered_features(test_df)

    train_df = add_log_features(train_df)
    test_df = add_log_features(test_df)

    scaler = ScoreScaler().fit(train_df, cols=SCORE_COLUMNS)

    train_scored = compute_score_with_scaler(train_df, scaler)
    test_scored = compute_score_with_scaler(test_df, scaler)

    train_scored = train_scored.dropna(subset=["ScorFinal"]).copy()
    test_scored = test_scored.dropna(subset=["ScorFinal"]).copy()

    threshold = float(train_scored["ScorFinal"].quantile(0.75))

    train_labeled = label_with_threshold(train_scored, threshold)
    test_labeled = label_with_threshold(test_scored, threshold)

    print(f"Threshold (q=0.75) computed on train: {threshold:.4f}")
    print("Train label distribution:\n", train_labeled["Merita"].value_counts(normalize=True))
    print("Test  label distribution:\n", test_labeled["Merita"].value_counts(normalize=True))

    full_system = Pipeline([
        ("preprocessor", build_preprocessing_pipeline()),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42))
    ])

    full_system.fit(train_labeled[MODEL_FEATURES], train_labeled["Merita"])

    pred = full_system.predict(test_labeled[MODEL_FEATURES])

    print("\nClassification report:\n", classification_report(test_labeled["Merita"], pred))
    print("Confusion matrix:\n", confusion_matrix(test_labeled["Merita"], pred))

    models_dir = PROJECT_ROOT / "Models"
    models_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "full_system": full_system,
        "threshold": threshold,
        "score_scaler": scaler
    }

    out_path = models_dir / "bundle_logreg_v1.joblib"
    joblib.dump(bundle, out_path)

    print(f"Model salvat în: {out_path}")

if __name__ == "__main__":
    main()