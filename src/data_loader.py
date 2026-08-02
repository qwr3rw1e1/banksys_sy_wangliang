"""Data loading and preprocessing pipeline."""

import logging
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    CATEGORICAL_FEATURES,
    CATEGORICAL_IMPUTE_STRATEGY,
    FEATURE_COLS,
    ID_COL,
    MODEL_DIR,
    NUMERIC_FEATURES,
    NUMERIC_IMPUTE_STRATEGY,
    PREPROCESSOR_FILE,
    RANDOM_SEED,
    TARGET_COL,
    TEST_SIZE,
    TRAIN_FILE,
)

logger = logging.getLogger(__name__)


def load_raw_data(filepath: Path | str | None = None) -> pd.DataFrame:
    """Load the raw bank marketing CSV file into a DataFrame.

    Args:
        filepath: Path to the CSV file. Defaults to TRAIN_FILE from config.

    Returns:
        DataFrame with all columns from the CSV.
    """
    path = Path(filepath) if filepath else TRAIN_FILE
    logger.info("Loading data from %s", path)
    df = pd.read_csv(path)
    logger.info("Loaded %d rows × %d columns", len(df), len(df.columns))
    return df


def get_feature_columns() -> list[str]:
    """Return the ordered list of feature column names."""
    return [c for c in FEATURE_COLS if c not in (TARGET_COL, ID_COL)]


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate features and target from a DataFrame.

    Args:
        df: DataFrame containing both features and the target column.

    Returns:
        (X, y) tuple — feature matrix and target Series.
    """
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    # Encode target as 0/1
    y = y.map({"no": 0, "yes": 1})
    if y.isna().any():
        logger.warning("Unexpected target values found: %s", y.unique())
    return X, y


def build_preprocessor() -> ColumnTransformer:
    """Build a sklearn ColumnTransformer for the feature matrix.

    Numeric features: median imputation + standardization.
    Categorical features: most-frequent imputation + one-hot encoding.

    Returns:
        Configured ColumnTransformer ready to fit/transform.
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=NUMERIC_IMPUTE_STRATEGY)),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=CATEGORICAL_IMPUTE_STRATEGY)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    return preprocessor


def preprocess_data(
    X: pd.DataFrame,
    preprocessor: ColumnTransformer | None = None,
    fit: bool = True,
) -> tuple[pd.DataFrame, ColumnTransformer]:
    """Preprocess features: impute, scale, and encode.

    Args:
        X: Raw feature DataFrame.
        preprocessor: Existing ColumnTransformer. If None, a new one is created.
        fit: Whether to fit the preprocessor (True for training, False for inference).

    Returns:
        (X_processed, preprocessor) — transformed feature array and the preprocessor.
    """
    if preprocessor is None:
        preprocessor = build_preprocessor()

    if fit:
        X_transformed = preprocessor.fit_transform(X)
    else:
        X_transformed = preprocessor.transform(X)

    logger.info("Preprocessed shape: %s", X_transformed.shape)
    return X_transformed, preprocessor


def split_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified train/test split.

    Args:
        X: Feature DataFrame.
        y: Target Series.
        test_size: Fraction for test split.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=RANDOM_SEED,
    )
    logger.info(
        "Train: %d rows, Test: %d rows (test_size=%.0f%%)",
        len(X_train), len(X_test), test_size * 100,
    )
    return X_train, X_test, y_train, y_test


def save_preprocessor(preprocessor: ColumnTransformer, filepath: Path | str | None = None) -> None:
    """Persist the fitted preprocessor to disk.

    Args:
        preprocessor: Fitted ColumnTransformer.
        filepath: Output path. Defaults to PREPROCESSOR_FILE.
    """
    path = Path(filepath) if filepath else PREPROCESSOR_FILE
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(preprocessor, f)
    logger.info("Preprocessor saved to %s", path)


def load_preprocessor(filepath: Path | str | None = None) -> ColumnTransformer:
    """Load a persisted preprocessor from disk.

    Args:
        filepath: Path to the pickle file. Defaults to PREPROCESSOR_FILE.

    Returns:
        The deserialized ColumnTransformer.
    """
    path = Path(filepath) if filepath else PREPROCESSOR_FILE
    with open(path, "rb") as f:
        preprocessor = pickle.load(f)
    logger.info("Preprocessor loaded from %s", path)
    return preprocessor
