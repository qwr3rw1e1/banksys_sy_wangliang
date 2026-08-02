# Stage 1: Build dependencies
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

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8888/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8888", "--server.address=0.0.0.0"]
