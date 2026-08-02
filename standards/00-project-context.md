# 00 — 项目上下文（Project Context）

## 项目名称

`banksys_sy_wangliang` — 银行营销数据分析与认购预测系统

## 项目概述

基于银行营销历史数据，构建一个 Web 应用，包含两大核心功能模块：
1. **数据分析交互页面** — 对银行营销数据进行多维度探索性分析与可视化
2. **在线预测系统** — 基于离线训练的机器学习模型，通过用户点选输入特征，实时预测客户是否会认购定期存款

## 技术栈

| 层面 | 选型 |
|------|------|
| 语言 | Python 3.11 |
| Web 框架 | Streamlit |
| 数据分析 | pandas、numpy、matplotlib、seaborn / plotly |
| 机器学习 | scikit-learn（分类模型：LogisticRegression、RandomForest、XGBoost 等） |
| 测试 | pytest |
| 代码检查 | ruff |
| 容器化 | Docker |
| 端口 | 8888 |

## 仓库信息

- 仓库名称：`banksys_sy_wangliang`
- Docker 镜像/容器名称：`banksys_sy_wangliang`
- 仓库类型：开源仓库
- 目标：跑通完整 CI + CD（持续集成 + 持续部署）

## 数据说明

数据位于 `data/` 目录下，为银行营销活动记录数据。典型字段包括：
- **客户基本信息**：年龄、职业、婚姻状况、教育水平、是否有违约记录、账户余额
- **客户资产与负债**：是否有住房贷款、是否有个人贷款
- **营销活动信息**：联系方式、联系日期（日/月）、通话时长
- **历史营销信息**：本次活动联系次数、上次联系间隔天数、之前联系次数、上次活动结果
- **目标变量**：客户是否认购定期存款（yes / no）

> 注：具体字段以 `data/` 目录下实际数据文件为准，实现前需先进行数据探查。

## 项目目录结构（规划）

```text
banksys_sy_wangliang/
├── data/                    # 原始数据目录
├── models/                  # 离线训练保存的模型文件
├── notebooks/               # EDA & 建模实验 Jupyter Notebook
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # 数据加载 & 预处理
│   ├── eda.py               # 探索性数据分析逻辑
│   ├── model_train.py       # 离线模型训练
│   ├── model_predict.py     # 在线预测逻辑
│   └── config.py            # 配置常量
├── app/
│   ├── __init__.py
│   ├── page_analysis.py     # 数据分析交互页面
│   └── page_prediction.py   # 在线预测页面
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_model_train.py
│   └── test_model_predict.py
├── app.py                   # Streamlit 主入口
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── pyproject.toml           # ruff + pytest 配置
├── .github/
│   └── workflows/
│       └── ci-cd.yml        # GitHub Actions CI/CD
├── standards/               # 项目规范文档（本目录）
└── PROGRESS.md              # 进度跟踪
```

## 核心约束

- 必须使用 Streamlit 作为 Web 框架
- 预测功能页面的输入方式为**点选表单**（下拉框、单选、滑块等），不支持自由文本输入
- 模型训练为**离线**过程，不集成到 Web 运行时
- 端口固定为 `8888`
- 全流程通过 Docker 容器化运行
