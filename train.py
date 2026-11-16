"""Train the final purchase-prediction model and save it to model.bin.

The model family and its hyper-parameters were selected in notebook.ipynb
(sections "Hyper-parameter tuning" and "Model selection"). This script repeats
only what is needed to produce the final model:

  1. load and prepare the data;
  2. hold out the same 20% test set as the notebook;
  3. estimate performance with 5-fold cross-validation on the other 80%
     and pick the decision threshold from the out-of-fold predictions;
  4. fit the model on the full 80% and evaluate it once on the test set;
  5. save the pipeline, the threshold and some metadata with pickle.

Usage:

    uv run python train.py
    uv run python train.py --data data/online_shoppers_intention.csv --output model.bin
"""

from __future__ import annotations

import argparse
import hashlib
import pickle
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline, make_pipeline

from features import COLUMN_RENAMES, FEATURES, TARGET, prepare_record

SEED = 1
TEST_SIZE = 0.2
N_SPLITS = 5
EXPECTED_SHAPE = (12330, 18)

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "data" / "online_shoppers_intention.csv"
DEFAULT_MODEL_PATH = PROJECT_DIR / "model.bin"

# Best configuration found by cross-validation in notebook.ipynb.
FINAL_MODEL_NAME = "HistGradientBoostingClassifier"
FINAL_PARAMS: dict[str, Any] = {
    "learning_rate": 0.05,
    "max_iter": 75,
    "max_leaf_nodes": 16,
    "min_samples_leaf": 50,
}


def load_data(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Read the raw CSV, rename the columns to snake_case and check its integrity."""
    df = pd.read_csv(path)
    if df.shape != EXPECTED_SHAPE:
        raise ValueError(f"unexpected data shape {df.shape}, expected {EXPECTED_SHAPE}")
    missing = set(COLUMN_RENAMES) - set(df.columns)
    if missing:
        raise ValueError(f"missing columns in the data file: {sorted(missing)}")
    df = df.rename(columns=COLUMN_RENAMES)
    if df.isna().any().any():
        raise ValueError("the data file contains missing values, which is not expected")
    df[TARGET] = df[TARGET].astype(int)
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified 80/20 split into (full_train, test). Same split as in the notebook."""
    df_full_train, df_test = train_test_split(
        df, test_size=TEST_SIZE, random_state=SEED, stratify=df[TARGET]
    )
    return df_full_train.reset_index(drop=True), df_test.reset_index(drop=True)


def to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Turn data-frame rows into the feature dictionaries used by the pipeline."""
    return [prepare_record(row) for row in df[FEATURES].to_dict(orient="records")]


def build_pipeline(params: dict[str, Any] | None = None) -> Pipeline:
    """DictVectorizer (one-hot encoding of the categorical features) + gradient boosting."""
    params = FINAL_PARAMS if params is None else params
    return make_pipeline(
        DictVectorizer(sparse=False),
        HistGradientBoostingClassifier(early_stopping=False, random_state=SEED, **params),
    )


def choose_threshold(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Threshold among 0.01, 0.02, ..., 0.99 that maximises the F1 score."""
    thresholds = np.round(np.arange(0.01, 1.00, 0.01), 2)
    f1_scores = [f1_score(y_true, probabilities >= threshold) for threshold in thresholds]
    return float(thresholds[int(np.argmax(f1_scores))])


def classification_report_at(y_true: np.ndarray, probabilities: np.ndarray, threshold: float) -> dict[str, float]:
    """Precision, recall and F1 of the decision `probability >= threshold`."""
    predictions = probabilities >= threshold
    tn, fp, fn, tp = confusion_matrix(y_true, predictions).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


def sha256_of(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def train(data_path: str | Path = DEFAULT_DATA_PATH, model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    print(f"Loading data from {data_path}")
    df = load_data(data_path)
    df_full_train, df_test = split_data(df)
    y_full_train = df_full_train[TARGET].to_numpy()
    y_test = df_test[TARGET].to_numpy()
    records_full_train = to_records(df_full_train)
    records_test = to_records(df_test)
    print(
        f"  {len(df):,} sessions, {df[TARGET].mean():.1%} end with a purchase | "
        f"train: {len(df_full_train):,} | test: {len(df_test):,}"
    )

    print(f"\n{N_SPLITS}-fold cross-validation on the training part")
    print(f"  model: {FINAL_MODEL_NAME} {FINAL_PARAMS}")
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(records_full_train))
    fold_auc, fold_ap = [], []
    for fold, (train_idx, val_idx) in enumerate(cv.split(records_full_train, y_full_train), start=1):
        fold_pipeline = build_pipeline()
        fold_pipeline.fit([records_full_train[i] for i in train_idx], y_full_train[train_idx])
        oof[val_idx] = fold_pipeline.predict_proba([records_full_train[i] for i in val_idx])[:, 1]
        fold_auc.append(roc_auc_score(y_full_train[val_idx], oof[val_idx]))
        fold_ap.append(average_precision_score(y_full_train[val_idx], oof[val_idx]))
        print(f"  fold {fold}: ROC AUC = {fold_auc[-1]:.4f} | average precision = {fold_ap[-1]:.4f}")
    cv_auc, cv_auc_std = float(np.mean(fold_auc)), float(np.std(fold_auc))
    cv_ap, cv_ap_std = float(np.mean(fold_ap)), float(np.std(fold_ap))
    print(f"  mean  : ROC AUC = {cv_auc:.4f} +- {cv_auc_std:.4f} | average precision = {cv_ap:.4f} +- {cv_ap_std:.4f}")

    threshold = choose_threshold(y_full_train, oof)
    print(f"\nDecision threshold (max F1 on out-of-fold predictions): {threshold:.2f}")

    print("\nTraining the final model on the full training part")
    pipeline = build_pipeline()
    pipeline.fit(records_full_train, y_full_train)

    test_probabilities = pipeline.predict_proba(records_test)[:, 1]
    test_auc = float(roc_auc_score(y_test, test_probabilities))
    test_ap = float(average_precision_score(y_test, test_probabilities))
    test_brier = float(brier_score_loss(y_test, test_probabilities))
    test_report = classification_report_at(y_test, test_probabilities, threshold)
    print("\nHeld-out test set")
    print(f"  ROC AUC           = {test_auc:.4f}")
    print(f"  average precision = {test_ap:.4f}   (a random ranking would score {y_test.mean():.4f})")
    print(f"  Brier score       = {test_brier:.4f}")
    print(
        f"  at threshold {threshold:.2f}: precision = {test_report['precision']:.4f} | "
        f"recall = {test_report['recall']:.4f} | F1 = {test_report['f1']:.4f}"
    )
    print(
        f"  confusion matrix: TP = {test_report['tp']} | FP = {test_report['fp']} | "
        f"FN = {test_report['fn']} | TN = {test_report['tn']}"
    )

    artifact = {
        "pipeline": pipeline,
        "threshold": threshold,
        "metadata": {
            "model": FINAL_MODEL_NAME,
            "params": dict(FINAL_PARAMS),
            "features": list(FEATURES),
            "target": TARGET,
            "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "data_sha256": sha256_of(data_path),
            "n_train": len(df_full_train),
            "n_test": len(df_test),
            "metrics": {
                "cv_roc_auc_mean": cv_auc,
                "cv_roc_auc_std": cv_auc_std,
                "cv_average_precision_mean": cv_ap,
                "cv_average_precision_std": cv_ap_std,
                "test_roc_auc": test_auc,
                "test_average_precision": test_ap,
                "test_brier": test_brier,
                "test_at_threshold": test_report,
            },
            "versions": {
                "python": platform.python_version(),
                "scikit-learn": sklearn.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
            },
        },
    }
    with open(model_path, "wb") as f_out:
        pickle.dump(artifact, f_out)
    print(f"\nModel saved to {model_path}")
    return artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the purchase-prediction model.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH), help="path to the CSV file with the sessions")
    parser.add_argument("--output", default=str(DEFAULT_MODEL_PATH), help="where to save the trained model")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(args.data, args.output)
