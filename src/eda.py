"""Exploratory data analysis logic — pure functions, no Streamlit dependency."""

import pandas as pd

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TARGET_COL


def dataset_overview(df: pd.DataFrame) -> dict:
    """Return high-level dataset metadata.

    Args:
        df: Raw DataFrame including the target column.

    Returns:
        Dict with keys: n_rows, n_cols, missing_pct, numeric_cols, categorical_cols.
    """
    n_rows, n_cols = df.shape
    missing_pct = round(df.isna().mean().mean() * 100, 2)
    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "missing_pct": missing_pct,
        "numeric_cols": len(NUMERIC_FEATURES),
        "categorical_cols": len(CATEGORICAL_FEATURES),
    }


def target_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Compute target variable distribution.

    Args:
        df: DataFrame with the target column.

    Returns:
        DataFrame with columns: class, count, pct.
    """
    counts = df[TARGET_COL].value_counts(dropna=False)
    dist = pd.DataFrame({
        "class": counts.index,
        "count": counts.values,
        "pct": (counts.values / counts.sum() * 100).round(1),
    })
    return dist.reset_index(drop=True)


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute descriptive statistics for numeric features.

    Args:
        df: DataFrame with numeric columns.

    Returns:
        DataFrame with describe() output transposed, including column name.
    """
    numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    if not numeric_cols:
        return pd.DataFrame()
    summary = df[numeric_cols].describe().T
    summary.index.name = "feature"
    return summary.reset_index()


def categorical_summary(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Compute cardinality and top-N frequency for each categorical feature.

    Args:
        df: DataFrame with categorical columns.
        top_n: Number of top categories to include.

    Returns:
        DataFrame with columns: feature, unique_count, top_category, top_count, top_pct.
    """
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    rows = []
    for col in cat_cols:
        series = df[col]
        unique = series.nunique(dropna=False)
        top_value = series.value_counts().index[0] if unique > 0 else None
        top_count = series.value_counts().iloc[0] if unique > 0 else 0
        top_pct = round(top_count / len(series) * 100, 1) if unique > 0 else 0.0
        rows.append({
            "feature": col,
            "unique_count": unique,
            "top_category": top_value,
            "top_count": top_count,
            "top_pct": top_pct,
        })
    return pd.DataFrame(rows)


def subscription_rate_by_category(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Compute subscription rate for each category of a given feature.

    Args:
        df: DataFrame with target and categorical features.
        feature: The categorical feature name.

    Returns:
        DataFrame with columns: category, total, subscribed, rate.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"DataFrame must contain '{TARGET_COL}' column")

    grouped = df.groupby(feature)[TARGET_COL].agg(["count", lambda s: (s == "yes").sum()])
    grouped.columns = ["total", "subscribed"]
    grouped["rate"] = (grouped["subscribed"] / grouped["total"] * 100).round(1)
    return grouped.sort_values("total", ascending=False).reset_index()


def numeric_by_target(df: pd.DataFrame, feature: str) -> pd.DataFrame:
    """Compute grouped statistics of a numeric feature split by target.

    Args:
        df: DataFrame with target column.
        feature: The numeric feature name.

    Returns:
        DataFrame with mean, median, std per target class.
    """
    if TARGET_COL not in df.columns:
        raise ValueError(f"DataFrame must contain '{TARGET_COL}' column")

    grouped = df.groupby(TARGET_COL)[feature].agg(["mean", "median", "std", "count"])
    return grouped.round(2).reset_index()


def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Compute correlation matrix for numeric features.

    Args:
        df: DataFrame with numeric columns.

    Returns:
        Square correlation DataFrame.
    """
    numeric_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    return df[numeric_cols].corr().round(3)
