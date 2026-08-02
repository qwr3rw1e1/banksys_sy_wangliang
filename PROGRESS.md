# PROGRESS — 项目进度跟踪

> 状态标记：⬜ 待开始 | 🔵 进行中 | ✅ 已完成 | ❌ 阻塞

---

## 阶段 0：项目初始化

| ID | 任务 | 状态 | 备注 |
|----|------|------|------|
| T-0.1 | 创建项目目录结构（src/、app/、tests/、models/、notebooks/） | ✅ | |
| T-0.2 | 编写 `requirements.txt`，锁定依赖版本 | ✅ | |
| T-0.3 | 编写 `pyproject.toml`，配置 ruff + pytest | ✅ | N 规则已忽略（sklearn 命名约定） |
| T-0.4 | 编写 `.gitignore` + `.dockerignore` | ✅ | |
| T-0.5 | 编写 `Dockerfile` + `docker-compose.yml` | ✅ | 多阶段构建，python:3.11-slim |
| T-0.6 | 编写 `README.md` | ✅ | |
| T-0.7 | 创建 `.github/workflows/ci-cd.yml` | ✅ | GHCR 推送 |
| T-0.8 | 初始化 git 仓库 + 首次提交 | ✅ | 已推送至 github.com/qwr3rw1e1/banksys_sy_wangliang |

---

## 阶段 1：数据层

| ID | 任务 | 状态 | 备注 |
|----|------|------|------|
| T-1.1 | 数据探查 — 确认字段、类型、缺失情况 | ✅ | train.csv (22,500×22), test.csv (7,500×21) |
| T-1.2 | 实现 `src/config.py` | ✅ | |
| T-1.3 | 实现 `src/data_loader.py` | ✅ | 预处理流水线：中位数填充 + 标准化 + OneHot |
| T-1.4 | 编写 `tests/test_data_loader.py` | ✅ | 12 个测试 |

---

## 阶段 2：数据分析页面

| ID | 任务 | 状态 | 备注 |
|----|------|------|------|
| T-2.1 | 实现 `src/eda.py` | ✅ | EDA 纯逻辑函数 |
| T-2.2 | 实现 `app/page_analysis.py` | ✅ | 数据概览 + 直方图/箱线图/柱状图/散点图/热力图 |
| T-2.3 | 实现 `app.py` — Streamlit 多页面导航 | ✅ | 侧边栏 📊 数据分析 / 🔮 在线预测 |

---

## 阶段 3：模型训练

| ID | 任务 | 状态 | 备注 |
|----|------|------|------|
| T-3.1 | 实现 `src/model_train.py` | ✅ | 三模型对比 + 选优 |
| T-3.2 | 运行训练脚本，生成模型到 `models/` | ✅ | XGBoost 最优 (AUC=0.8899) |
| T-3.3 | 编写 `tests/test_model_train.py` | ✅ | 8 个测试 |

### 模型训练结果

| 模型 | AUC-ROC | F1 | Accuracy |
|------|---------|-----|----------|
| XGBoost ⭐ | **0.8899** | 0.4648 | 0.8869 |
| Random Forest | 0.8842 | 0.5129 | 0.8696 |
| Logistic Regression | 0.8089 | 0.4789 | 0.7944 |

---

## 阶段 4：在线预测页面

| ID | 任务 | 状态 | 备注 |
|----|------|------|------|
| T-4.1 | 实现 `src/model_predict.py` | ✅ | 模型加载 + 单条推理 |
| T-4.2 | 实现 `app/page_prediction.py` | ✅ | 点选表单 + 仪表盘结果展示 |
| T-4.3 | 编写 `tests/test_model_predict.py` | ✅ | 5 个测试 |

---

## 阶段 5：整合测试 & CI/CD 验证

| ID | 任务 | 状态 | 备注 |
|----|------|------|------|
| T-5.1 | 全量 `pytest` 通过 | ✅ | 27/27 通过，覆盖率 72% |
| T-5.2 | 全量 `ruff check` 零告警 | ✅ | |
| T-5.3 | Docker 构建 + 启动验证 | ⬜ | 需本地 Docker 环境 |
| T-5.4 | Streamlit 启动验证 | ✅ | HTTP 200，两个页面正常 |
| T-5.5 | CI/CD 流水线 GitHub Actions | ✅ | ✅ lint-test + build-push 均已通过 |
| T-5.6 | GHCR 镜像推送 | ✅ | ghcr.io/qwr3rw1e1/banksys_sy_wangliang |

---

## 验证摘要

```text
✅ ruff check .     — All checks passed!
✅ pytest           — 27 passed in 4.45s
✅ coverage         — 72% (≥70% target)
✅ model training   — XGBoost AUC=0.8899
✅ streamlit app    — http://localhost:8888 → 200
⬜ git init + push  — pending
⬜ docker build     — pending (need Docker)
```
