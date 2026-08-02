"""Model loading and inference — pure logic, no Streamlit dependency."""

import logging
import pickle
from pathlib import Path

import pandas as pd

from src.config import (
    FEATURE_COLS,
    MODEL_FILE,
)

logger = logging.getLogger(__name__)


def load_model(filepath: Path | str | None = None):
    """Load the trained model from disk.

    Args:
        filepath: Path to model pickle. Defaults to MODEL_FILE.

    Returns:
        The deserialized model instance.
    """
    path = Path(filepath) if filepath else MODEL_FILE
    with open(path, "rb") as f:
        model = pickle.load(f)
    logger.info("Model loaded from %s", path)
    return model


def predict(
    model,
    preprocessor,
    input_data: dict,
) -> tuple[int, float]:
    """Make a single prediction from a dict of raw feature values.

    Args:
        model: Trained classifier with predict_proba.
        preprocessor: Fitted ColumnTransformer.
        input_data: Dict[str, value] mapping feature names to raw values.

    Returns:
        (prediction, probability) — prediction is 0 or 1, probability is for class 1.
    """
    # Build a single-row DataFrame in the correct column order
    row = {col: input_data.get(col) for col in FEATURE_COLS}
    df = pd.DataFrame([row])

    # Preprocess
    X = preprocessor.transform(df)

    # Predict
    proba = model.predict_proba(X)[0, 1]
    pred = int(proba >= 0.5)

    logger.info("Prediction: %d (probability=%.4f)", pred, proba)
    return pred, round(float(proba), 4)


def get_feature_choices(df: pd.DataFrame) -> dict[str, list]:
    """Extract unique valid values for each categorical feature from training data.

    Used to populate dropdown options in the prediction form.

    Args:
        df: Training DataFrame containing all feature columns.

    Returns:
        Dict[feature_name, list_of_unique_sorted_values] for categorical features.
    """
    from src.config import CATEGORICAL_FEATURES

    choices = {}
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            vals = sorted(df[col].dropna().unique().tolist())
            choices[col] = vals
    return choices
