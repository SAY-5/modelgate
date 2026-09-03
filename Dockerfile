FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY modelgate ./modelgate
RUN uv sync --frozen --no-dev

COPY artifacts ./artifacts

EXPOSE 8000
ENV MODELGATE_ARTIFACTS_DIR=/app/artifacts
CMD ["uv", "run", "--no-sync", "uvicorn", "modelgate.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]
