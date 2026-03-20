import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


from Src.config import MODEL_FEATURES, SCORE_COLUMNS, PROCESSED_DIR
from Src.io import load_skincare_dv
from Src.preprocessing import build_preprocessing_pipeline
from Src.scoring import (
    add_log_features,
    ScoreScaler,
    compute_score_with_scaler,
    label_with_threshold
)
from Src.feature_engineering import add_engineered_features


"""
Script experimental pentru compararea mai multor algoritmi de clasificare
în aceleași condiții de preprocessing și etichetare.

Modelul oficial al aplicației rămâne Logistic Regression.
Acest script este folosit doar pentru analiză comparativă.
"""

RANDOM_STATE = 42
TEST_SIZE = 0.2


def prepare_train_test_data():
    df = load_skincare_dv()

    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
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

    return train_labeled, test_labeled


def build_model_pipeline(model):
    return Pipeline([
        ("preprocessor", build_preprocessing_pipeline()),
        ("classifier", model)
    ])


def evaluate_model(model_name, pipeline, train_df, test_df):
    X_train = train_df[MODEL_FEATURES]
    y_train = train_df["Merita"]

    X_test = test_df[MODEL_FEATURES]
    y_test = test_df["Merita"]

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n=== {model_name} ===")
    print(f"Features folosite: {MODEL_FEATURES}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print("Confusion matrix:")
    print(cm)

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }


def main():
    train_labeled, test_labeled = prepare_train_test_data()

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE
        ),
        "Decision Tree": DecisionTreeClassifier(
            random_state=RANDOM_STATE,
            max_depth=None
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE
        )
    }

    results = []

    for model_name, model in models.items():
        pipeline = build_model_pipeline(model)
        result = evaluate_model(
            model_name,
            pipeline,
            train_labeled,
            test_labeled
        )
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="F1", ascending=False)

    print("\n=== REZUMAT COMPARATIV ALGORITMI (EXPERIMENTAL) ===")
    print(results_df.to_string(index=False))

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / "algorithm_comparison.csv"
    results_df.to_csv(out_path, index=False)

    print(f"\nRezultatele comparative au fost salvate în: {out_path}")
    print("Modelul oficial al aplicației rămâne Logistic Regression.")
    print("Acest script este folosit doar pentru comparație experimentală între algoritmi.")


if __name__ == "__main__":
    main()