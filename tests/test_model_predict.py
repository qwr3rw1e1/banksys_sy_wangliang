"""Tests for model_predict module."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from src.data_loader import build_preprocessor, split_features_target
from src.model_predict import get_feature_choices, load_model, predict


class TestPredict:
    """Tests for predict function."""

    def test_returns_int_and_float(self, sample_df):
        # Train a tiny model on the sample data
        X, y = split_features_target(sample_df)
        pp = build_preprocessor()
        X_pp = pp.fit_transform(X)

        model = LogisticRegression(random_state=42)
        model.fit(X_pp, y)

        input_data = {
            "age": 35,
            "duration": 200,
            "campaign": 1,
            "pdays": 999,
            "previous": 0,
            "emp_var_rate": 0.0,
            "cons_price_index": 94.0,
            "cons_conf_index": -40.0,
            "lending_rate3m": 1.5,
            "nr_employed": 5000.0,
            "job": "admin.",
            "marital": "married",
            "education": "university.degree",
            "default": "no",
            "housing": "yes",
            "loan": "no",
            "contact": "cellular",
            "month": "may",
            "day_of_week": "mon",
            "poutcome": "nonexistent",
        }

        pred, proba = predict(model, pp, input_data)

        assert pred in (0, 1)
        assert 0.0 <= proba <= 1.0

    def test_prediction_is_binary(self, sample_df):
        X, y = split_features_target(sample_df)
        pp = build_preprocessor()
        X_pp = pp.fit_transform(X)

        model = LogisticRegression(random_state=42)
        model.fit(X_pp, y)

        input_data = {
            "age": 60,
            "duration": 500,
            "campaign": 5,
            "pdays": 3,
            "previous": 2,
            "emp_var_rate": -1.0,
            "cons_price_index": 93.0,
            "cons_conf_index": -42.0,
            "lending_rate3m": 3.0,
            "nr_employed": 4900.0,
            "job": "technician",
            "marital": "single",
            "education": "high.school",
            "default": "unknown",
            "housing": "no",
            "loan": "yes",
            "contact": "telephone",
            "month": "jun",
            "day_of_week": "tue",
            "poutcome": "failure",
        }

        pred, proba = predict(model, pp, input_data)
        assert pred == int(proba >= 0.5)


class TestLoadModel:
    """Tests for load_model."""

    def test_loads_pickled_model(self, tmp_path):
        model = LogisticRegression(random_state=42)
        model.fit(np.random.rand(10, 3), np.array([0, 1] * 5))

        import pickle
        path = tmp_path / "test_model.pkl"
        with open(path, "wb") as f:
            pickle.dump(model, f)

        loaded = load_model(filepath=path)
        assert loaded is not None
        assert hasattr(loaded, "predict_proba")


class TestGetFeatureChoices:
    """Tests for get_feature_choices."""

    def test_returns_dict_with_categorical_keys(self, sample_df):
        choices = get_feature_choices(sample_df)

        assert isinstance(choices, dict)
        assert "job" in choices
        assert "marital" in choices
        assert "education" in choices
        # Each value should be a list
        assert isinstance(choices["job"], list)
        assert len(choices["job"]) > 0

    def test_numeric_features_not_in_choices(self, sample_df):
        choices = get_feature_choices(sample_df)
        assert "age" not in choices
        assert "duration" not in choices
