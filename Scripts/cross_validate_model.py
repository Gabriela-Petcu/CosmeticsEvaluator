import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from Src.config import PROCESSED_DIR, SCORE_COLUMNS, MODEL_FEATURES
from Src.io import load_skincare_dv
from Src.preprocessing import build_preprocessing_pipeline
from Src.scoring import (
    add_log_features,
    ScoreScaler,
    compute_score_with_scaler,
    label_with_threshold
)
from Src.feature_engineering import add_engineered_features


RANDOM_STATE = 42
N_SPLITS = 5

def main():
    df = load_skincare_dv().reset_index(drop=True)

    # Pentru StratifiedKFold avem nevoie de o etichetă inițială de stratificare.
    # Aceasta este folosită doar pentru împărțirea relativ echilibrată a datelor,
    # nu pentru evaluarea finală pe folduri.
    strat_df = add_engineered_features(df.copy())
    strat_df = add_log_features(strat_df)

    strat_scaler = ScoreScaler().fit(strat_df, cols=SCORE_COLUMNS)
    strat_df = compute_score_with_scaler(strat_df, strat_scaler)
    strat_df = strat_df.dropna(subset=["ScorFinal"]).copy()

    strat_threshold = float(strat_df["ScorFinal"].quantile(0.75))
    strat_df = label_with_threshold(strat_df, strat_threshold)

    valid_indices = strat_df.index.to_numpy()
    df_valid = df.loc[valid_indices].reset_index(drop=True)
    y_strat = strat_df["Merita"].reset_index(drop=True)

    cv = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(df_valid, y_strat), start=1):
        train_raw = df_valid.iloc[train_idx].copy()
        val_raw = df_valid.iloc[val_idx].copy()

        # 1. Pregătire train fold
        train_prepared = add_engineered_features(train_raw)
        train_prepared = add_log_features(train_prepared)

        # 2. Fit ScoreScaler DOAR pe train fold
        fold_scaler = ScoreScaler().fit(train_prepared, cols=SCORE_COLUMNS)

        # 3. Calcul scor baseline pe train și val cu scaler-ul din train
        train_scored = compute_score_with_scaler(train_prepared, fold_scaler)
        val_prepared = add_engineered_features(val_raw)
        val_prepared = add_log_features(val_prepared)
        val_scored = compute_score_with_scaler(val_prepared, fold_scaler)

        # 4. Elimină rândurile fără ScorFinal
        train_scored = train_scored.dropna(subset=["ScorFinal"]).copy()
        val_scored = val_scored.dropna(subset=["ScorFinal"]).copy()

        # 5. Prag calculat DOAR pe train fold
        fold_threshold = float(train_scored["ScorFinal"].quantile(0.75))

        # 6. Etichetare train și val cu pragul din train fold
        train_labeled = label_with_threshold(train_scored, fold_threshold)
        val_labeled = label_with_threshold(val_scored, fold_threshold)

        # 7. Seturi pentru ML
        X_train = train_labeled[MODEL_FEATURES].copy()
        y_train = train_labeled["Merita"].copy()

        X_val = val_labeled[MODEL_FEATURES].copy()
        y_val = val_labeled["Merita"].copy()

        # 8. Pipeline oficial
        full_system = Pipeline([
            ("preprocessor", build_preprocessing_pipeline()),
            ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE))
        ])

        full_system.fit(X_train, y_train)
        y_pred = full_system.predict(X_val)

        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        fold_results.append({
            "fold": fold_idx,
            "train_size": len(train_labeled),
            "val_size": len(val_labeled),
            "threshold": fold_threshold,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    folds_df = pd.DataFrame(fold_results)

    summary_df = pd.DataFrame({
        "metric": ["accuracy", "precision", "recall", "f1"],
        "mean": [
            folds_df["accuracy"].mean(),
            folds_df["precision"].mean(),
            folds_df["recall"].mean(),
            folds_df["f1"].mean(),
        ],
        "std": [
            folds_df["accuracy"].std(ddof=0),
            folds_df["precision"].std(ddof=0),
            folds_df["recall"].std(ddof=0),
            folds_df["f1"].std(ddof=0),
        ],
    })

    print("=== CROSS-VALIDATION RESULTS (5-FOLD, leakage-free) ===")
    print("\nPer-fold results:")
    print(folds_df.to_string(index=False))

    print("\nSummary:")
    print(summary_df.to_string(index=False))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    folds_path = PROCESSED_DIR / "cross_validation_folds.csv"
    summary_path = PROCESSED_DIR / "cross_validation_summary.csv"

    folds_df.to_csv(folds_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print(f"\nPer-fold results saved to: {folds_path}")
    print(f"Summary results saved to: {summary_path}")


if __name__ == "__main__":
    main()