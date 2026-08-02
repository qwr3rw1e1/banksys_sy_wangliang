"""Central configuration constants for the project."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
TRAIN_FILE = DATA_DIR / "train.csv"
TEST_FILE = DATA_DIR / "test.csv"
MODEL_FILE = MODEL_DIR / "best_model.pkl"
PREPROCESSOR_FILE = MODEL_DIR / "preprocessor.pkl"
METRICS_FILE = MODEL_DIR / "metrics.json"

# ── Data ───────────────────────────────────────────────
TARGET_COL = "subscribe"
ID_COL = "id"
TEST_SIZE = 0.2
RANDOM_SEED = 42

# Numeric features (excluding id and target)
NUMERIC_FEATURES: list[str] = [
    "age",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "emp_var_rate",
    "cons_price_index",
    "cons_conf_index",
    "lending_rate3m",
    "nr_employed",
]

# Categorical features
CATEGORICAL_FEATURES: list[str] = [
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
    "contact",
    "month",
    "day_of_week",
    "poutcome",
]

# All feature columns (in a fixed order for the prediction form)
FEATURE_COLS: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ── Preprocessing ──────────────────────────────────────
CAT_HIGH_CARDINALITY_THRESHOLD = 10  # Use LabelEncoding if ≤ this, else OneHot
NUMERIC_IMPUTE_STRATEGY = "median"
CATEGORICAL_IMPUTE_STRATEGY = "most_frequent"

# ── Model ──────────────────────────────────────────────
MODEL_CANDIDATES: list[str] = ["logistic", "random_forest", "xgboost"]

MODEL_PARAMS: dict = {
    "logistic": {
        "max_iter": 2000,
        "class_weight": "balanced",
    },
    "random_forest": {
        "n_estimators": 200,
        "max_depth": 15,
        "class_weight": "balanced",
    },
    "xgboost": {
        "n_estimators": 200,
        "max_depth": 6,
        "learning_rate": 0.1,
        "eval_metric": "logloss",
    },
}

# ── App display ────────────────────────────────────────
FEATURE_LABELS: dict[str, str] = {
    "age": "年龄",
    "duration": "通话时长 (秒)",
    "campaign": "本次活动联系次数",
    "pdays": "上次联系间隔天数",
    "previous": "之前联系次数",
    "emp_var_rate": "就业变化率",
    "cons_price_index": "消费者物价指数",
    "cons_conf_index": "消费者信心指数",
    "lending_rate3m": "3个月贷款利率",
    "nr_employed": "雇员人数",
    "job": "职业",
    "marital": "婚姻状况",
    "education": "教育水平",
    "default": "是否有违约记录",
    "housing": "是否有住房贷款",
    "loan": "是否有个人贷款",
    "contact": "联系方式",
    "month": "最后联系月份",
    "day_of_week": "最后联系星期",
    "poutcome": "上次活动结果",
}
