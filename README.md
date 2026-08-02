# banksys_sy_wangliang

银行营销数据分析与认购预测系统 — 基于 Streamlit 的 Web 应用。

## 功能

| 功能 | 说明 |
|------|------|
| 📊 数据分析 | 银行营销数据多维度交互式探索与可视化 |
| 🔮 在线预测 | 基于机器学习模型的客户认购意向实时预测 |

## 技术栈

- **Python 3.11** + **Streamlit**（Web 框架）
- **scikit-learn** + **XGBoost**（机器学习）
- **pandas** + **plotly**（数据分析与可视化）
- **pytest** + **ruff**（测试与代码检查）
- **Docker**（容器化部署）
- **GitHub Actions**（CI/CD）

## 快速启动

### 本地开发

```bash
# 安装依赖
pip install -r requirements.txt

# 运行应用
streamlit run app.py
# 访问 http://localhost:8888
```

### Docker

```bash
# 构建并启动
docker-compose up -d

# 或手动构建
docker build -t banksys_sy_wangliang .
docker run -p 8888:8888 banksys_sy_wangliang
```

### 模型训练

```bash
python src/model_train.py
# 模型保存至 models/
```

### 测试

```bash
pytest
ruff check .
```

## 项目结构

```text
src/            # 核心逻辑（数据加载、EDA、训练、预测）
app/            # Streamlit 页面
data/           # 银行营销数据集
models/         # 训练好的模型文件
tests/          # 单元测试
standards/      # 项目规范文档
```

## CI/CD

![CI/CD Pipeline](https://github.com/<user>/banksys_sy_wangliang/actions/workflows/ci-cd.yml/badge.svg)

- push / PR → ruff lint → pytest → Docker build → push to GHCR

## License

MIT
