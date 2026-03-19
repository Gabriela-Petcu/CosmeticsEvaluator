import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from Src.io import load_skincare_dv
from Src.preprocessing import build_preprocessing_pipeline
from Src.scoring import add_log_features, ScoreScaler, compute_score_with_scaler, label_with_threshold
from Src.feature_engineering import add_engineered_features

BASE_LOG_FEATURES = [
    "n_of_reviews",
    "n_of_loves"
]

BASE_STANDARD_FEATURES = [
    "review_score",
    "price_per_ounce"
]

ENGINEERED_LOG_FEATURES = [
    "n_of_reviews",
    "n_of_loves"
]

ENGINEERED_STANDARD_FEATURES = [
    "review_score",
    "price_per_ounce",
    "popularity_score",
    "engagement_score",
    "value_score",
    "review_strength"
]

BASE_FEATURES = [
    "n_of_reviews",
    "n_of_loves",
    "review_score",
    "price_per_ounce"
]

ENGINEERED_FEATURES = [
    "n_of_reviews",
    "n_of_loves",
    "review_score",
    "price_per_ounce",
    "popularity_score",
    "engagement_score",
    "value_score",
    "review_strength"
]


def evaluate_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_list: list[str],
    log_features: list[str],
    standard_features: list[str],
    model_name: str
):
    model = Pipeline([
        (
            "preprocessor",
            build_preprocessing_pipeline(
                log_feature_columns=log_features,
                standard_feature_columns=standard_features
            )
        ),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    model.fit(train_df[feature_list], train_df["Merita"])
    pred = model.predict(test_df[feature_list])

    accuracy = accuracy_score(test_df["Merita"], pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        test_df["Merita"],
        pred,
        average="binary",
        pos_label=1,
        zero_division=0
    )
    cm = confusion_matrix(test_df["Merita"], pred)

    print(f"\n=== {model_name} ===")
    print(f"Features folosite: {feature_list}")
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

    scaler = ScoreScaler().fit(train_df, cols=["review_score", "log_reviews", "log_loves", "price_per_ounce"])

    train_scored = compute_score_with_scaler(train_df, scaler)
    test_scored = compute_score_with_scaler(test_df, scaler)

    threshold = float(train_scored["ScorFinal"].quantile(0.75))

    train_labeled = label_with_threshold(train_scored, threshold)
    test_labeled = label_with_threshold(test_scored, threshold)

    old_results = evaluate_model(
        train_labeled,
        test_labeled,
        BASE_FEATURES,
        BASE_LOG_FEATURES,
        BASE_STANDARD_FEATURES,
        "RandomForest - Model vechi"
    )

    new_results = evaluate_model(
        train_labeled,
        test_labeled,
        ENGINEERED_FEATURES,
        ENGINEERED_LOG_FEATURES,
        ENGINEERED_STANDARD_FEATURES,
        "RandomForest - Model nou"
    )

    results_df = pd.DataFrame([old_results, new_results])

    print("\n=== COMPARAȚIE FINALĂ ===")
    print(results_df)


if __name__ == "__main__":
    main()