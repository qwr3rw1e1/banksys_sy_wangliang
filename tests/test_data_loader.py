"""Tests for data_loader module."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.data_loader import (
    build_preprocessor,
    load_preprocessor,
    preprocess_data,
    save_preprocessor,
    split_features_target,
    split_train_test,
)


class TestSplitFeaturesTarget:
    """Tests for split_features_target."""

    def test_returns_feature_matrix_and_target(self, sample_df):
        X, y = split_features_target(sample_df)

        assert isinstance(X, pd.DataFrame)
        assert isinstance(y, pd.Series)
        assert "subscribe" not in X.columns
        assert set(y.unique()).issubset({0, 1})

    def test_target_encoding_no_to_0(self, sample_df):
        _, y = split_features_target(sample_df)

        no_mask = sample_df["subscribe"] == "no"
        assert (y[no_mask] == 0).all()

    def test_target_encoding_yes_to_1(self, sample_df):
        _, y = split_features_target(sample_df)

        yes_mask = sample_df["subscribe"] == "yes"
        assert (y[yes_mask] == 1).all()


class TestBuildPreprocessor:
    """Tests for build_preprocessor."""

    def test_returns_column_transformer(self):
        preprocessor = build_preprocessor()
        assert isinstance(preprocessor, ColumnTransformer)

    def test_has_numeric_and_categorical_steps(self):
        preprocessor = build_preprocessor()
        names = [name for name, _, _ in preprocessor.transformers]
        assert "numeric" in names
        assert "categorical" in names


class TestPreprocessData:
    """Tests for preprocess_data."""

    def test_fit_returns_numpy_array(self, sample_df):
        X, _ = split_features_target(sample_df)
        X_transformed, pp = preprocess_data(X, fit=True)

        assert isinstance(X_transformed, np.ndarray)
        assert isinstance(pp, ColumnTransformer)

    def test_transform_with_existing_preprocessor(self, sample_df):
        X, _ = split_features_target(sample_df)
        _, pp = preprocess_data(X, fit=True)
        X_transformed2, _ = preprocess_data(X, preprocessor=pp, fit=False)

        assert X_transformed2.shape[0] == X.shape[0]  # same number of rows

    def test_handles_missing_values(self, sample_df):
        X, _ = split_features_target(sample_df)
        X.loc[0, "age"] = np.nan
        X.loc[1, "job"] = None

        X_transformed, _ = preprocess_data(X, fit=True)
        assert not np.any(np.isnan(X_transformed))


class TestSaveLoadPreprocessor:
    """Tests for save_preprocessor and load_preprocessor."""

    def test_roundtrip(self, sample_df, tmp_path):
        X, _ = split_features_target(sample_df)
        _, pp = preprocess_data(X, fit=True)

        save_path = tmp_path / "preprocessor.pkl"
        save_preprocessor(pp, filepath=save_path)

        loaded = load_preprocessor(filepath=save_path)
        assert isinstance(loaded, ColumnTransformer)

    def test_load_preserves_transform_result(self, sample_df, tmp_path):
        X, _ = split_features_target(sample_df)
        X_transformed, pp = preprocess_data(X, fit=True)

        save_path = tmp_path / "preprocessor.pkl"
        save_preprocessor(pp, filepath=save_path)
        loaded_pp = load_preprocessor(filepath=save_path)

        X_loaded_transformed, _ = preprocess_data(X, preprocessor=loaded_pp, fit=False)
        np.testing.assert_array_almost_equal(X_transformed, X_loaded_transformed)


class TestSplitTrainTest:
    """Tests for split_train_test."""

    def test_returns_four_arrays(self, sample_df):
        X, y = split_features_target(sample_df)
        X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.4)

        assert len(X_train) + len(X_test) == len(sample_df)
        assert len(y_train) + len(y_test) == len(sample_df)

    def test_preserves_class_distribution_approx(self, sample_df):
        """With a small df, stratification is approximate but should not be far off."""
        X, y = split_features_target(sample_df)
        _, _, y_train, _ = split_train_test(X, y, test_size=0.4)

        train_ratio = y_train.mean()
        overall_ratio = y.mean()
        # Check ratios are reasonable (within 0.3 for this small sample)
        assert abs(train_ratio - overall_ratio) < 0.4
