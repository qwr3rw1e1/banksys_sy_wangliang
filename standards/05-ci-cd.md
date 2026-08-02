# 05 — CI/CD 规范（CI/CD Standards）

## 工作流触发条件

| 事件 | 触发 |
|------|------|
| `push` 到 `main` 分支 | ✅ |
| `pull_request` 到 `main` 分支 | ✅ |
| 手动触发 (`workflow_dispatch`) | ✅ |

## 流水线阶段

```text
Checkout → Setup Python → Install Deps → Lint → Test → Build Docker → (Push Image)
                                                       │
                                                       └── 仅 main 分支 push 时推送镜像
```

## GitHub Actions 工作流文件

文件路径：`.github/workflows/ci-cd.yml`

### 关键配置

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]
  workflow_dispatch:

env:
  PYTHON_VERSION: "3.11"
  DOCKER_IMAGE: banksys_sy_wangliang

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Install dependencies
        run: pip install -r requirements.txt
      - name: Lint with ruff
        run: ruff check .
      - name: Test with pytest
        run: pytest --cov=src --cov-report=term-missing

  build-and-push:
    needs: lint-and-test
    runs-on: ubuntu-latest
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker image
        run: docker build -t ${{ env.DOCKER_IMAGE }} .
      - name: Push to registry
        run: |
          # 推送至 Docker Hub 或 GHCR
          ...
```

## 分支策略

- `main` — 主分支，保持可部署状态
- `feature/*` — 功能分支，合并前需通过 CI

## 质量门禁

| 门禁 | 要求 |
|------|------|
| Ruff | 零告警 |
| Pytest | 全部通过 |
| 覆盖率 | ≥ 70%（以 `src/` 为统计范围） |

> 以上三项任一失败，CI 标记为失败，阻断 CD 阶段。

## Commit 规范

推荐使用 Conventional Commits：

```text
feat: 新功能
fix: 修复
docs: 文档
test: 测试
chore: 构建/工具
refactor: 重构
```
