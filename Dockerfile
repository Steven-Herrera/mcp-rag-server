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

COPY --chown=appuser:appuser pyproject.toml uv.lock ./

# Runs as root so uv can write to the venv.
RUN uv sync --frozen --no-dev --no-editable \
    && rm -rf ~/.cache/uv

# Pre-download model weights only — snapshot_download is faster than
# instantiating the full SentenceTransformer (no warmup pass).
RUN /app/.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2')"

USER appuser

COPY --chown=appuser:appuser mcp_server/ mcp_server/

EXPOSE 9000

ENTRYPOINT ["tini", "--"]
CMD ["/app/.venv/bin/python", "-m", "mcp_server.main"]