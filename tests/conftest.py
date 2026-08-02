"""Test fixtures and helpers."""

import pytest


@pytest.fixture
def sample_df():
    """A small synthetic bank marketing DataFrame for testing."""
    import pandas as pd

    return pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "age": [30, 45, 28, 52, 38],
        "job": ["admin.", "technician", "services", "management", "admin."],
        "marital": ["married", "single", "married", "divorced", "single"],
        "education": [
            "university.degree", "high.school", "professional.course",
            "university.degree", "high.school",
        ],
        "default": ["no", "no", "no", "unknown", "no"],
        "housing": ["yes", "yes", "no", "yes", "no"],
        "loan": ["no", "yes", "no", "no", "yes"],
        "contact": ["cellular", "telephone", "cellular", "cellular", "telephone"],
        "month": ["may", "jun", "jul", "aug", "may"],
        "day_of_week": ["mon", "tue", "wed", "thu", "fri"],
        "duration": [120, 300, 450, 89, 210],
        "campaign": [1, 2, 1, 3, 1],
        "pdays": [999, 3, 999, 10, 999],
        "previous": [0, 1, 0, 2, 0],
        "poutcome": ["nonexistent", "failure", "nonexistent", "success", "nonexistent"],
        "emp_var_rate": [1.4, -0.5, 1.1, -1.8, 0.3],
        "cons_price_index": [93.0, 94.2, 92.8, 93.5, 94.0],
        "cons_conf_index": [-36.0, -40.1, -42.0, -38.5, -35.8],
        "lending_rate3m": [1.2, 1.5, 1.3, 4.0, 1.1],
        "nr_employed": [5000.0, 5100.0, 5050.0, 4950.0, 5000.0],
        "subscribe": ["no", "yes", "no", "yes", "no"],
    })
