"""Tests for model_train module."""

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from src.model_train import (
    evaluate_model,
    get_model,
    save_metrics,
    save_model,
    train_and_compare,
)


class TestGetModel:
    """Tests for get_model factory."""

    def test_logistic_returns_lr(self):
        model = get_model("logistic", random_state=42)
        assert isinstance(model, LogisticRegression)

    def test_random_forest_returns_rf(self):
        model = get_model("random_forest", random_state=42)
        assert isinstance(model, RandomForestClassifier)

    def test_xgboost_returns_xgb(self):
        model = get_model("xgboost", random_state=42)
        from xgboost import XGBClassifier
        assert isinstance(model, XGBClassifier)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown model_type"):
            get_model("nonexistent")

    def test_all_candidates_return_models(self):
        from src.config import MODEL_CANDIDATES
        for name in MODEL_CANDIDATES:
            model = get_model(name, random_state=42)
            assert model is not None
            assert hasattr(model, "fit")


class TestEvaluateModel:
    """Tests for evaluate_model."""

    def test_returns_all_metric_keys(self):
        X = np.random.rand(10, 5)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        model = LogisticRegression(random_state=42)
        model.fit(X, y)

        metrics = evaluate_model(model, X, y)
        for key in ("accuracy", "precision", "recall", "f1", "auc_roc"):
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0

    def test_perfect_predictions(self):
        """All metrics should be 1.0 for a perfect classifier on separable data."""
        X = np.array([[10, 10], [10, 10], [-10, -10], [-10, -10]])
        y = np.array([1, 1, 0, 0])
        model = LogisticRegression(random_state=42)
        model.fit(X, y)
        metrics = evaluate_model(model, X, y)
        assert metrics["accuracy"] == 1.0
        assert metrics["auc_roc"] == 1.0


class TestSaveModel:
    """Tests for save_model."""

    def test_saves_file(self, tmp_path):
        model = LogisticRegression(random_state=42).fit(
            np.random.rand(10, 3), np.array([0, 1] * 5),
        )
        path = tmp_path / "test_model.pkl"
        save_model(model, filepath=path)
        assert path.exists()


class TestSaveMetrics:
    """Tests for save_metrics."""

    def test_creates_valid_json(self, tmp_path):
        results = {
            "logistic": {"accuracy": 0.9, "auc_roc": 0.88},
            "random_forest": {"accuracy": 0.92, "auc_roc": 0.91},
        }
        path = tmp_path / "metrics.json"
        save_metrics(results, "random_forest", filepath=path)
        assert path.exists()

        import json
        with open(path) as f:
            data = json.load(f)
        assert data["best_model"] == "random_forest"
        assert "trained_at" in data


class TestTrainAndCompare:
    """Tests for train_and_compare."""

    def test_returns_results_and_model(self):
        X_train = np.random.rand(20, 5)
        X_test = np.random.rand(10, 5)
        y_train = np.array([0, 1] * 10)
        y_test = np.array([0, 1] * 5)

        results, best_model = train_and_compare(X_train, X_test, y_train, y_test)

        assert len(results) >= 3
        for name in ("logistic", "random_forest", "xgboost"):
            assert name in results
            assert "auc_roc" in results[name]
        assert best_model is not None
