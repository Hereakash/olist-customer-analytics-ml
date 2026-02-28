import os
import json
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)

import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv("config/settings.env")

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")

TRAIN_SNAPSHOT_DATE = "2018-06-01"

MODEL_DIR = "models"
REPORT_DIR = "reports"
BUNDLE_PATH = os.path.join(MODEL_DIR, "churn_torch_bundle.joblib")
METRICS_PATH = os.path.join(MODEL_DIR, "churn_torch_metrics.json")
HISTORY_CSV = os.path.join(MODEL_DIR, "churn_torch_history.csv")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


class NumpyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)  # logits


@torch.no_grad()
def predict_proba(model, loader, device):
    model.eval()
    probs = []
    ys = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p)
        ys.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(ys)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))

    engine = create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    )

    df = pd.read_sql(
        text("""
            SELECT
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

    y = df["will_reorder_90d"].astype(int).to_numpy()
    X = df.drop(columns=["will_reorder_90d"])

    numeric_features = [
        "recency_days",
        "frequency",
        "monetary",
        "avg_delivery_days",
        "late_rate",
        "avg_review_score",
    ]
    categorical_features = ["customer_state"]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_features,
            ),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    X_tr_t = preprocessor.fit_transform(X_tr)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    # MLP requires dense arrays
    if hasattr(X_tr_t, "toarray"):
        X_tr_t = X_tr_t.toarray()
        X_val_t = X_val_t.toarray()
        X_test_t = X_test_t.toarray()

    input_dim = X_tr_t.shape[1]
    model = MLP(input_dim).to(device)

    # imbalance: pos_weight for BCEWithLogitsLoss
    pos = int(y_tr.sum())
    neg = int((y_tr == 0).sum())
    if pos == 0:
        raise RuntimeError("No positive samples in train split. Use earlier cutoff date.")
    pos_weight = torch.tensor([neg / pos], device=device, dtype=torch.float32)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    batch_size = 4096
    train_loader = DataLoader(NumpyDataset(X_tr_t, y_tr), batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(NumpyDataset(X_val_t, y_val), batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(NumpyDataset(X_test_t, y_test), batch_size=batch_size, shuffle=False, num_workers=0)

    best_val_pr_auc = -1.0
    best_state = None
    patience = 8
    bad_epochs = 0

    history = []
    epochs = 50

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        train_loss = float(np.mean(losses))

        val_proba, y_val_true = predict_proba(model, val_loader, device)
        val_roc_auc = roc_auc_score(y_val_true, val_proba)
        val_pr_auc = average_precision_score(y_val_true, val_proba)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_roc_auc": float(val_roc_auc),
                "val_pr_auc": float(val_pr_auc),
            }
        )

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.5f} | val_roc_auc={val_roc_auc:.4f} | val_pr_auc={val_pr_auc:.4f}"
        )

        # early stopping
        if val_pr_auc > best_val_pr_auc + 1e-6:
            best_val_pr_auc = val_pr_auc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping: no val_pr_auc improvement for {patience} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # test evaluation
    test_proba, y_test_true = predict_proba(model, test_loader, device)
    test_roc_auc = roc_auc_score(y_test_true, test_proba)
    test_pr_auc = average_precision_score(y_test_true, test_proba)

    precisions, recalls, thresholds = precision_recall_curve(y_test_true, test_proba)
    f1 = (2 * precisions * recalls) / (precisions + recalls + 1e-12)
    best_idx = int(np.nanargmax(f1))
    best_threshold = float(thresholds[max(best_idx - 1, 0)]) if len(thresholds) else 0.5

    y_pred = (test_proba >= best_threshold).astype(int)
    cm = confusion_matrix(y_test_true, y_pred)

    # save history
    hist_df = pd.DataFrame(history)
    hist_df.to_csv(HISTORY_CSV, index=False)

    # plots
    plt.figure(figsize=(8, 5))
    plt.plot(hist_df["epoch"], hist_df["train_loss"], label="train_loss")
    plt.title("PyTorch NN Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "torch_nn_loss.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(hist_df["epoch"], hist_df["val_pr_auc"], label="val_pr_auc")
    plt.title("PyTorch NN Validation PR-AUC")
    plt.xlabel("Epoch")
    plt.ylabel("PR-AUC")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "torch_nn_val_pr_auc.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix (thr={best_threshold:.3f})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "torch_nn_confusion_matrix.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(recalls, precisions)
    plt.title("Precision-Recall Curve (PyTorch NN)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "torch_nn_pr_curve.png"), dpi=160)
    plt.close()

    # save weights + bundle
    torch_model_path = os.path.join(MODEL_DIR, "churn_torch_nn.pt")
    torch.save(model.state_dict(), torch_model_path)

    joblib.dump(
        {
            "preprocessor": preprocessor,
            "model_state_dict_path": torch_model_path,
            "input_dim": input_dim,
        },
        BUNDLE_PATH,
    )

    metrics = {
        "train_snapshot_date": TRAIN_SNAPSHOT_DATE,
        "rows": int(len(df)),
        "positives": int(y.sum()),
        "positive_rate": float(y.mean()),
        "device": str(device),
        "pos_weight": float(pos_weight.detach().cpu().numpy()[0]),
        "best_val_pr_auc": float(best_val_pr_auc),
        "test_roc_auc": float(test_roc_auc),
        "test_pr_auc": float(test_pr_auc),
        "best_threshold_f1": float(best_threshold),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(y_test_true, y_pred, output_dict=True, zero_division=0),
        "artifacts": {
            "bundle": BUNDLE_PATH,
            "weights": torch_model_path,
            "history_csv": HISTORY_CSV,
            "reports": [
                "reports/torch_nn_loss.png",
                "reports/torch_nn_val_pr_auc.png",
                "reports/torch_nn_pr_curve.png",
                "reports/torch_nn_confusion_matrix.png",
            ],
        },
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Saved weights:", torch_model_path)
    print("Saved bundle:", BUNDLE_PATH)
    print("Saved metrics:", METRICS_PATH)
    print("TEST ROC-AUC:", test_roc_auc, "TEST PR-AUC:", test_pr_auc, "Best thr:", best_threshold)


if __name__ == "__main__":
    main()