FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CAPTIX_MODEL_DIR=/app/models \
    CAPTIX_METRICS_ENABLED=false \
    ORT_INTRA_OP_THREADS=2

WORKDIR /app
COPY deploy/requirements-runtime.txt /tmp/requirements-runtime.txt
RUN python -m pip install --no-cache-dir --only-binary=:all: --require-hashes -r /tmp/requirements-runtime.txt \
    && rm /tmp/requirements-runtime.txt

COPY src/captix_runtime /app/captix_runtime
COPY service /app/service
COPY models/model.onnx models/manifest.json /app/models/
# Fail the build for missing/corrupt artifacts; startup also verifies before loading.
RUN python -c "from pathlib import Path; from captix_runtime.artifacts import load_verified_manifest; load_verified_manifest(Path('/app/models'))"

USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=5s --timeout=3s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health/ready', timeout=2).close()"]
CMD ["python", "-m", "uvicorn", "service.main:create_app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-graceful-shutdown", "30", "--no-access-log"]
