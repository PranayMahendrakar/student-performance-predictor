"""
Student Performance Predictor - Model Training
Trains Random Forest and Gradient Boosting classifiers to predict student performance.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    roc_auc_score, f1_score
)
import joblib
import os
import json
from datetime import datetime


FEATURE_COLUMNS = [
    "attendance_rate",
    "assignment_avg",
    "midterm_score",
    "study_hours_per_week",
    "participation_score",
    "previous_gpa",
    "assignment_completion",
    "quiz_avg",
]

TARGET_COLUMN = "performance"


def generate_dataset(n_samples=2000, random_state=42):
    rng = np.random.RandomState(random_state)
    data = {}
    engaged = rng.choice([True, False], size=n_samples, p=[0.65, 0.35])
    data["attendance_rate"] = np.where(
        engaged,
        np.clip(rng.normal(85, 10, n_samples), 60, 100),
        np.clip(rng.normal(50, 15, n_samples), 10, 75),
    )
    data["study_hours_per_week"] = np.clip(
        data["attendance_rate"] / 10 + rng.normal(0, 2, n_samples), 0, 20
    )
    data["assignment_avg"] = np.clip(
        0.5 * data["attendance_rate"] + 0.3 * data["study_hours_per_week"] * 2
        + rng.normal(0, 8, n_samples), 0, 100,
    )
    data["assignment_completion"] = np.clip(
        data["attendance_rate"] * 0.9 + rng.normal(0, 5, n_samples), 0, 100
    )
    data["midterm_score"] = np.clip(
        0.4 * data["assignment_avg"] + 0.4 * data["study_hours_per_week"] * 3
        + rng.normal(0, 10, n_samples), 0, 100,
    )
    data["quiz_avg"] = np.clip(
        0.6 * data["assignment_avg"] + rng.normal(0, 7, n_samples), 0, 100
    )
    data["participation_score"] = np.clip(
        data["attendance_rate"] / 12 + rng.normal(0, 1.5, n_samples), 0, 10
    )
    data["previous_gpa"] = np.clip(
        data["assignment_avg"] / 25 + rng.normal(0, 0.4, n_samples), 0, 4.0
    )
    df = pd.DataFrame(data)
    composite = (
        df["attendance_rate"] * 0.20 + df["assignment_avg"] * 0.25
        + df["midterm_score"] * 0.25 + df["study_hours_per_week"] * 1.5
        + df["previous_gpa"] * 5 + df["quiz_avg"] * 0.10
        + df["assignment_completion"] * 0.10 + df["participation_score"] * 1.0
    )
    p33, p66 = np.percentile(composite, [33, 66])
    df[TARGET_COLUMN] = pd.cut(
        composite, bins=[-np.inf, p33, p66, np.inf],
        labels=["Fail", "At Risk", "Pass"],
    )
    return df


def train_models(df):
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5,
        min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1,
    )
    rf.fit(X_train_s, y_train)

    gbm = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=5,
        min_samples_split=5, min_samples_leaf=2, subsample=0.8, random_state=42,
    )
    gbm.fit(X_train_s, y_train)

    results = {}
    for name, model in [("random_forest", rf), ("gradient_boosting", gbm)]:
        preds = model.predict(X_test_s)
        proba = model.predict_proba(X_test_s)
        results[name] = {
            "accuracy": accuracy_score(y_test, preds),
            "f1": f1_score(y_test, preds, average="weighted"),
            "auc": roc_auc_score(y_test, proba, multi_class="ovr"),
            "cv_accuracy": cross_val_score(model, X_train_s, y_train, cv=5).mean(),
            "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
            "feature_importances": dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist())),
            "report": classification_report(y_test, preds, target_names=le.classes_, output_dict=True),
        }
    return rf, gbm, scaler, le, results


def save_artifacts(rf, gbm, scaler, le, results, out_dir="artifacts"):
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(rf, os.path.join(out_dir, "random_forest.pkl"))
    joblib.dump(gbm, os.path.join(out_dir, "gradient_boosting.pkl"))
    joblib.dump(scaler, os.path.join(out_dir, "scaler.pkl"))
    joblib.dump(le, os.path.join(out_dir, "label_encoder.pkl"))
    meta = {
        "trained_at": datetime.utcnow().isoformat(),
        "features": FEATURE_COLUMNS,
        "target": TARGET_COLUMN,
        "classes": le.classes_.tolist(),
        "metrics": {k: {m: v for m, v in results[k].items()
                        if m in ("accuracy","f1","auc","cv_accuracy")}
                    for k in results},
    }
    with open(os.path.join(out_dir, "model_metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Artifacts saved to '{out_dir}/'")


if __name__ == "__main__":
    print("Generating dataset ...")
    df = generate_dataset(n_samples=3000)
    print(f"Dataset shape: {df.shape}")
    rf, gbm, scaler, le, results = train_models(df)
    save_artifacts(rf, gbm, scaler, le, results)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/student_data.csv", index=False)
    print("Done!")

