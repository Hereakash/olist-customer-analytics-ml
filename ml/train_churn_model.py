import os
import json
import joblib
import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    classification_report,
    confusion_matrix,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

load_dotenv("config/settings.env")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

TRAIN_SNAPSHOT_DATE = "2018-06-01"  # training cutoff we built
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "churn_model.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "churn_metrics.json")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    engine = create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    df = pd.read_sql(
        text("""
            SELECT
              customer_unique_id,
              recency_days,
              frequency,
              monetary,
              avg_delivery_days,
              late_rate,
              avg_review_score,
              customer_state,
              will_reorder_90d
            FROM mart.customer_churn_ml_dataset
            WHERE snapshot_date = :d
        """),
        engine,
        params={"d": TRAIN_SNAPSHOT_DATE},
    )

    if df.empty:
        raise RuntimeError(f"No training data found for snapshot_date={TRAIN_SNAPSHOT_DATE}")

    y = df["will_reorder_90d"].astype(int)
    X = df.drop(columns=["will_reorder_90d", "customer_unique_id"])

    numeric_features = [
        "recency_days",
        "frequency",
        "monetary",
        "avg_delivery_days",
        "late_rate",
        "avg_review_score",
    ]
    categorical_features = ["customer_state"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Stratified split because of imbalance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    models = {
        "logreg": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            n_jobs=None,
            solver="lbfgs",
        ),
        "rf": RandomForestClassifier(
            n_estimators=400,
            random_state=42,
            class_weight="balanced_subsample",
            n_jobs=-1,
            max_depth=None,
            min_samples_leaf=5,
        ),
    }

    results = {}
    best_name = None
    best_pr_auc = -1
    best_pipe = None

    for name, clf in models.items():
        pipe = Pipeline(steps=[("preprocess", preprocessor), ("model", clf)])
        pipe.fit(X_train, y_train)

        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)

        roc_auc = roc_auc_score(y_test, proba)
        pr_auc = average_precision_score(y_test, proba)
        cm = confusion_matrix(y_test, pred)

        results[name] = {
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(y_test, pred, output_dict=True, zero_division=0),
        }

        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_name = name
            best_pipe = pipe

    # Save best model
    joblib.dump(best_pipe, MODEL_PATH)

    out = {
        "train_snapshot_date": TRAIN_SNAPSHOT_DATE,
        "best_model": best_name,
        "best_pr_auc": float(best_pr_auc),
        "results": results,
        "positive_rate": float(y.mean()),
        "rows": int(len(df)),
        "positives": int(y.sum()),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Saved model:", MODEL_PATH)
    print("Saved metrics:", METRICS_PATH)
    print("Best model:", best_name, "PR-AUC:", best_pr_auc)
    print("Rows:", len(df), "Positives:", int(y.sum()), "Positive rate:", y.mean())


if __name__ == "__main__":
    main()