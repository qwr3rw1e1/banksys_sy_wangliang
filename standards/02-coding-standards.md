# 02 — 编码规范（Coding Standards）

## Python 版本

项目统一使用 **Python 3.11**。

## 代码格式化与 Lint

使用 **ruff** 作为唯一的 lint & format 工具。

### `pyproject.toml` 配置参考

```toml
[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "F",   # pyflakes
    "I",   # isort
    "N",   # pep8-naming
    "W",   # pycodestyle warnings
    "UP",  # pyupgrade
]
ignore = [
    "E501", # line-too-long, 交由 formatter 处理
]

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
```

## 命名约定

| 类型 | 规范 | 示例 |
|------|------|------|
| 模块/文件 | `snake_case` | `data_loader.py` |
| 类 | `PascalCase` | `ModelTrainer` |
| 函数/方法 | `snake_case` | `load_data()` |
| 变量 | `snake_case` | `feature_cols` |
| 常量 | `UPPER_SNAKE_CASE` | `RANDOM_SEED` |
| 私有函数/变量 | 前缀 `_` | `_validate_input()` |

## 类型注解

- 所有公共函数必须有类型注解
- 使用 Python 3.11 内置类型：`list[X]` 而非 `typing.List[X]`

```python
def load_data(filepath: str) -> pd.DataFrame:
    """Load bank marketing data from CSV file."""
    ...
```

## Docstring

- 采用 Google 风格 docstring
- 公共函数必须有 docstring（一句话概述 + Args + Returns）
- 复杂逻辑加行内注释，注释使用英文

```python
def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    model_type: str = "random_forest",
) -> BaseEstimator:
    """Train a classification model on the given training data.

    Args:
        X_train: Feature matrix for training.
        y_train: Target labels for training.
        model_type: One of "logistic", "random_forest", "xgboost".

    Returns:
        Trained sklearn-compatible model instance.
    """
    ...
```

## 导入顺序

- 标准库 → 第三方库 → 本地模块
- 每组之间空一行
- 使用 `import X` 而非 `from X import *`

## 禁止事项

- 禁止使用 `print()` 输出日志，使用 `logging` 模块
- 禁止硬编码路径/魔术数字，统一放入 `config.py`
- 禁止在函数内部修改全局变量
- 禁止 `except:` 裸捕获，必须指定异常类型
