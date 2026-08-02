# 04 — 测试规范（Testing Standards）

## 测试框架

使用 **pytest** 作为唯一的测试框架。

## `pyproject.toml` 配置参考

```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = "test_*.py"
python_classes = "Test*"
python_functions = "test_*"
addopts = [
    "-v",
    "--strict-markers",
    "--tb=short",
    "--cov=src",
    "--cov-report=term-missing",
    "--cov-report=html",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks tests as integration tests",
]
```

## 测试目录结构

```text
tests/
├── __init__.py
├── conftest.py              # 共享 fixtures
├── test_data_loader.py      # 数据加载 & 预处理测试
├── test_model_train.py      # 模型训练测试
└── test_model_predict.py    # 模型预测测试
```

## 测试覆盖率目标

| 模块 | 目标覆盖率 |
|------|-----------|
| `src/*.py` | ≥ 70%（整体） |
| `src/config.py` | ≥ 90% |

## 测试编写规范

### 每个测试函数：

1. **Arrange** — 准备测试数据和依赖
2. **Act** — 执行被测函数
3. **Assert** — 验证结果

```python
def test_load_data_returns_dataframe(sample_csv_path):
    # Arrange
    # Act
    df = load_data(sample_csv_path)
    # Assert
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
```

### Fixtures

- 共享 fixture 放入 `conftest.py`
- 使用小型合成数据集而非真实大数据
- 对文件 I/O 使用 `tmp_path` fixture

```python
# conftest.py
@pytest.fixture
def sample_df():
    """Create a small synthetic bank marketing DataFrame."""
    return pd.DataFrame({
        "age": [30, 45, 28, 52],
        "job": ["admin.", "technician", "services", "management"],
        "y": ["no", "yes", "no", "yes"],
    })
```

### 测试命名

| 场景 | 命名模式 | 示例 |
|------|---------|------|
| 正常路径 | `test_<function>_<condition>` | `test_predict_returns_probability` |
| 边界情况 | `test_<function>_<edge_case>` | `test_load_data_handles_empty_file` |
| 异常情况 | `test_<function>_raises_<exception>` | `test_predict_raises_on_missing_model` |

## 禁止事项

- 禁止测试之间相互依赖（测试隔离）
- 禁止硬编码绝对路径
- 禁止在测试中调用真实外部 API 或网络请求
- 禁止提交未通过的测试
