# 06 — Docker & 部署规范（Docker & Deployment Standards）

## Dockerfile 设计

### 多阶段构建（推荐）

```dockerfile
# Stage 1: Build
FROM python:3.11-slim AS builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Stage 2: Runtime
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH="/root/.local/bin:${PATH}"
ENV STREAMLIT_SERVER_PORT=8888
ENV STREAMLIT_SERVER_HEADLESS=true

EXPOSE 8888

CMD ["streamlit", "run", "app.py", "--server.port=8888", "--server.address=0.0.0.0"]
```

### 约束

- 基础镜像：`python:3.11-slim`（最小化攻击面与体积）
- 应用端口：`8888`
- 必须以非 root 用户运行（生产环境）
- 不包含 `.venv`、`__pycache__`、`tests/`、`notebooks/`
- 模型文件（`models/`）包含在镜像中（离线训练后生成）

### .dockerignore

```text
__pycache__
*.pyc
.venv
.git
.gitignore
*.md
!README.md
tests/
notebooks/
.pytest_cache
.ruff_cache
```

## docker-compose.yml

```yaml
version: "3.9"

services:
  app:
    build: .
    image: banksys_sy_wangliang:latest
    container_name: banksys_sy_wangliang
    ports:
      - "8888:8888"
    environment:
      - STREAMLIT_SERVER_PORT=8888
      - STREAMLIT_SERVER_HEADLESS=true
    restart: unless-stopped
```

## 本地运行命令

```bash
# 构建
docker build -t banksys_sy_wangliang .

# 启动
docker run -p 8888:8888 banksys_sy_wangliang

# 或使用 compose
docker-compose up -d

# 访问
# http://localhost:8888
```

## Streamlit 配置

通过环境变量或 `.streamlit/config.toml` 统一管理：

```toml
[server]
port = 8888
headless = true
runOnSave = false

[browser]
gatherUsageStats = false
```

> 优先使用环境变量覆盖，方便容器化传参。

## 模型文件策略

- 训练脚本 (`src/model_train.py`) 生成模型文件到 `models/`
- 模型文件（`.pkl`）随镜像打包，确保容器自包含
- `models/` 目录不加入 `.gitignore`（或不加入 git 但打包进 Docker）
  > 决策：模型文件体积 ≤ 10MB 时打入镜像；否则采用 volume 挂载

## 健康检查

```dockerfile
HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8888/_stcore/health')"
```
