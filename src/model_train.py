"""Offline model training: multi-model comparison, selection, and persistence."""

import json
import logging
import pickle
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier

from src.config import (
    METRICS_FILE,
    MODEL_CANDIDATES,
    MODEL_DIR,
    MODEL_FILE,
    MODEL_PARAMS,
    RANDOM_SEED,
)
from src.data_loader import (
    load_raw_data,
    preprocess_data,
    save_preprocessor,
    split_features_target,
    split_train_test,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def get_model(model_type: str, random_state: int = RANDOM_SEED):
    """Factory: return an untrained model instance by name.

    Args:
        model_type: One of MODEL_CANDIDATES.
        random_state: Random seed for reproducibility.

    Returns:
        An sklearn-compatible classifier instance.
    """
    params = MODEL_PARAMS.get(model_type, {})
    params["random_state"] = random_state

    if model_type == "logistic":
        return LogisticRegression(**params)
    elif model_type == "random_forest":
        return RandomForestClassifier(**params)
    elif model_type == "xgboost":
        params.pop("eval_metric", None)
        return XGBClassifier(**params)
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Expected one of {MODEL_CANDIDATES}")


def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Compute evaluation metrics for a trained model.

    Args:
        model: Trained classifier with predict and predict_proba.
        X_test: Test feature matrix.
        y_test: Test target labels (0/1).

    Returns:
        Dict with keys: accuracy, precision, recall, f1, auc_roc.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "auc_roc": round(roc_auc_score(y_test, y_proba), 4),
    }


def train_and_compare(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
) -> tuple[dict, object]:
    """Train all candidate models and select the best by AUC-ROC.

    Args:
        X_train, X_test: Training/test feature matrices.
        y_train, y_test: Training/test labels.

    Returns:
        (results, best_model) — dict of model_name → metrics, and the best model instance.
    """
    results = {}
    best_model = None
    best_auc = -1.0
    best_name = ""

    for name in MODEL_CANDIDATES:
        logger.info("Training %s ...", name)
        model = get_model(name)
        model.fit(X_train, y_train)
        metrics = evaluate_model(model, X_test, y_test)
        results[name] = metrics
        logger.info(
            "  %s — AUC: %.4f | F1: %.4f | Acc: %.4f",
            name, metrics["auc_roc"], metrics["f1"], metrics["accuracy"],
        )

        if metrics["auc_roc"] > best_auc:
            best_auc = metrics["auc_roc"]
            best_model = model
            best_name = name

    logger.info("Best model: %s (AUC=%.4f)", best_name, best_auc)
    return results, best_model


def save_model(model, filepath: Path | str | None = None) -> None:
    """Persist trained model to disk.

    Args:
        model: Trained classifier.
        filepath: Output path. Defaults to MODEL_FILE.
    """
    path = Path(filepath) if filepath else MODEL_FILE
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Model saved to %s", path)


def save_metrics(results: dict, best_model_name: str, filepath: Path | str | None = None) -> None:
    """Persist evaluation results to JSON.

    Args:
        results: Dict of model_name → metrics.
        best_model_name: Name of the selected model.
        filepath: Output path. Defaults to METRICS_FILE.
    """
    path = Path(filepath) if filepath else METRICS_FILE
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "trained_at": datetime.now(UTC).isoformat(),
        "best_model": best_model_name,
        "results": results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("Metrics saved to %s", path)


def main() -> None:
    """Run the full training pipeline.

    Steps:
        1. Load raw data
        2. Split X/y, then train/test
        3. Preprocess (fit on train, transform both)
        4. Train and compare models
        5. Save best model, preprocessor, and metrics
    """
    logger.info("=" * 50)
    logger.info("Starting training pipeline")

    # 1. Load
    df = load_raw_data()

    # 2. Split
    X, y = split_features_target(df)
    X_train, X_test, y_train, y_test = split_train_test(X, y)

    # 3. Preprocess
    X_train_pp, preprocessor = preprocess_data(X_train, fit=True)
    X_test_pp, _ = preprocess_data(X_test, preprocessor=preprocessor, fit=False)

    # 4. Train
    results, best_model = train_and_compare(X_train_pp, X_test_pp, y_train, y_test)
    best_name = max(results, key=lambda k: results[k]["auc_roc"])

    # 5. Persist
    save_model(best_model)
    save_preprocessor(preprocessor)
    save_metrics(results, best_name)

    logger.info("Training pipeline completed successfully")


if __name__ == "__main__":
    main()
