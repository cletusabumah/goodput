# CPU-only image for Compose worker nodes (ticket 2.4).
# Build context is the repo root (see docker/compose.yaml).
FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY pyproject.toml requirements.txt README.md ./
COPY src ./src
# CPU torch index first so Linux images do not pull CUDA wheels.
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
      -r requirements.txt \
 && pip install --no-cache-dir .

COPY experiments ./experiments

CMD ["goodput-run", "--config", "experiments/compose.yaml"]
