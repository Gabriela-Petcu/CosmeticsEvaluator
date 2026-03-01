import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix

from Src.io import load_skincare_dv
from Src.preprocessing import build_preprocessing_pipeline
from Src.scoring import add_log_features, ScoreScaler, compute_score_with_scaler, label_with_threshold

df = load_skincare_dv()

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

train_df = add_log_features(train_df)
test_df = add_log_features(test_df)

score_cols = ["review_score", "log_reviews", "log_loves", "price_per_ounce"]
scaler = ScoreScaler().fit(train_df, cols=score_cols)

train_scored = compute_score_with_scaler(train_df, scaler)
test_scored = compute_score_with_scaler(test_df, scaler)

threshold = float(train_scored["ScorFinal"].quantile(0.75))

train_labeled = label_with_threshold(train_scored, threshold)
test_labeled = label_with_threshold(test_scored, threshold)

print(f"Threshold (q=0.75) computed on train: {threshold:.4f}")
print("Train label distribution:\n", train_labeled["Merita"].value_counts(normalize=True))
print("Test  label distribution:\n", test_labeled["Merita"].value_counts(normalize=True))

log_cols = ["n_of_reviews", "n_of_loves"]
std_cols = ["review_score", "price"]
features = log_cols + std_cols

full_system = Pipeline([
    ("preprocessor", build_preprocessing_pipeline(log_cols, std_cols)),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
])

full_system.fit(train_labeled[features], train_labeled["Merita"])

pred = full_system.predict(test_labeled[features])
print("\nClassification report:\n", classification_report(test_labeled["Merita"], pred))
print("Confusion matrix:\n", confusion_matrix(test_labeled["Merita"], pred))

Path("Models").mkdir(parents=True, exist_ok=True)

bundle = {
    "full_system": full_system,
    "threshold": threshold,
    "score_scaler": scaler
}

joblib.dump(bundle, "Models/bundle_v1.joblib")
print("✅ Models/bundle_v1.joblib salvat.")