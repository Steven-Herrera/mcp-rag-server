# syntax=docker/dockerfile:1
FROM python:3.12-slim AS base

WORKDIR /app

# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends tini \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

RUN useradd --create-home appuser \
    && chown appuser:appuser /app

USER appuser

COPY --chown=appuser:appuser pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-editable \
    && rm -rf ~/.cache/uv

COPY --chown=appuser:appuser mcp_server/ mcp_server/

# Pre-download the embedding model at build time so startup is fast.
RUN /app/.venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')"

EXPOSE 9000

# tini runs as PID 1 so it can forward signals (SIGTERM) to the app and reap zombie processes.
# Without it, Python as PID 1 may not shut down cleanly when Kubernetes stops the pod.
ENTRYPOINT ["tini", "--"]
CMD ["/app/.venv/bin/python", "-m", "mcp_server.main"]