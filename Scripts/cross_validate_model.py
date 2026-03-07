import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from Src.config import PROCESSED_DIR, SCORE_COLUMNS, MODEL_FEATURES
from Src.io import load_skincare_dv
from Src.preprocessing import build_preprocessing_pipeline
from Src.scoring import add_log_features, ScoreScaler, compute_score_with_scaler, label_with_threshold


def main():
    # 1. Load data
    df = load_skincare_dv()

    # 2. Build baseline labels
    df = add_log_features(df)

    scaler = ScoreScaler().fit(df, cols=SCORE_COLUMNS)

    df_scored = compute_score_with_scaler(df, scaler)
    threshold = float(df_scored["ScorFinal"].quantile(0.75))
    df_labeled = label_with_threshold(df_scored, threshold)

    # 3. Features and target
    X = df_labeled[MODEL_FEATURES].copy()
    y = df_labeled["Merita"].copy()

    # 4. Build ML pipeline
    full_system = Pipeline([
        ("preprocessor", build_preprocessing_pipeline()),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    # 5. Cross-validation setup
    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1"
    }

    # 6. Run cross-validation
    results = cross_validate(
        estimator=full_system,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        return_train_score=False
    )

    # 7. Build detailed results
    folds_df = pd.DataFrame({
        "fold": range(1, len(results["test_accuracy"]) + 1),
        "accuracy": results["test_accuracy"],
        "precision": results["test_precision"],
        "recall": results["test_recall"],
        "f1": results["test_f1"],
    })

    summary_df = pd.DataFrame({
        "metric": ["accuracy", "precision", "recall", "f1"],
        "mean": [
            np.mean(results["test_accuracy"]),
            np.mean(results["test_precision"]),
            np.mean(results["test_recall"]),
            np.mean(results["test_f1"]),
        ],
        "std": [
            np.std(results["test_accuracy"]),
            np.std(results["test_precision"]),
            np.std(results["test_recall"]),
            np.std(results["test_f1"]),
        ],
    })

    # 8. Print results
    print("=== CROSS-VALIDATION RESULTS (5-FOLD) ===")
    print("\nPer-fold results:")
    print(folds_df.to_string(index=False))

    print("\nSummary:")
    print(summary_df.to_string(index=False))

    # 9. Save results
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    folds_path = PROCESSED_DIR / "cross_validation_folds.csv"
    summary_path = PROCESSED_DIR / "cross_validation_summary.csv"

    folds_df.to_csv(folds_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\n✅ Per-fold results saved to: {folds_path}")
    print(f"✅ Summary results saved to: {summary_path}")


if __name__ == "__main__":
    main()