"""HTTP upload endpoint for document ingestion.

Provides a POST /ingest endpoint that accepts file uploads,
parses them, chunks the text, and upserts into Qdrant.  This is
mounted as a sub-application alongside the MCP Streamable HTTP
transport so both share the same port.

This endpoint is intended for admin/CLI use and is protected by the
edge-gw-auth gateway's API key or bearer token

Functions:
    _get_store: Returns VectorStore instance
    handle_ingest: Parses, chunks, then uploads text to the vector store
    health_check: Check if vector store is healthy
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from loguru import logger
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from mcp_server.chunking import chunk_text
from mcp_server.parsers import parse_file
from mcp_server.vectorstore import VectorStore

_store: VectorStore | None = None


def _get_store() -> VectorStore:
    """Return the singleton VectorStore if it is not yet instantiated

    Returns:
        _store: VectorStore instance
    """
    global _store  # noqa: PLW0603
    if _store is None:
        _store = VectorStore()
        _store.ensure_collection()
    return _store


async def handle_ingest(request: Request) -> JSONResponse:
    """Handle a file upload for ingestion.

    Accepts multipart/form-data with a file field and an
    optional source field to override the source path.

    Args:
        request: Incoming Starlette request.

    Returns:
        JSON response with ingestion results.
    """
    form = await request.form()
    upload = form.get("file")
    if upload is None:
        return JSONResponse({"error": "No file field in upload"}, status_code=400)

    source_override = form.get("source")

    # Write to a temp file so parsers can read it.
    suffix = Path(upload.filename or "unknown").suffix
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await upload.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        result = parse_file(tmp_path)
        if result is None:
            return JSONResponse(
                {"error": f"Unsupported file type: {suffix}"},
                status_code=400,
            )

        text, metadata = result
        source = source_override or upload.filename or "upload"
        metadata["original_filename"] = upload.filename or "unknown"

        chunks = chunk_text(text, source=source, metadata=metadata)
        store = _get_store()
        count = store.ingest_chunks(chunks)
        logger.info("HTTP ingest: {} chunks from '{}'", count, source)

        return JSONResponse(
            {
                "status": "ok",
                "source": source,
                "chunks_ingested": count,
            }
        )
    finally:
        tmp_path.unlink(missing_ok=True)


async def handle_health(request: Request) -> JSONResponse:
    """Health check verifying Qdrant connectivity and embedding model.

    Starlette requires request as the first argument, but it's not
    actually used here.

    Args:
        request: Incoming request

    Returns:
        JSON with status, qdrant, and embedding_model fields.
        Returns 503 if any dependency is unhealthy.
    """
    health: dict[str, str] = {}

    try:
        store = _get_store()
        info = store.collection_info()
        health["qdrant"] = f"ok ({info.points_count} points)"
    except Exception as exc:
        health["qdrant"] = f"error: {exc}"

    try:
        from mcp_server.embeddings import embed_query as _eq  # noqa: PLC0415

        vec = _eq("health check")
        health["embedding_model"] = f"ok (dim={len(vec)})"
    except Exception as exc:
        health["embedding_model"] = f"error: {exc}"

    all_ok = all(v.startswith("ok") for v in health.values())
    health["status"] = "ok" if all_ok else "degraded"
    status_code = 200 if all_ok else 503
    return JSONResponse(health, status_code=status_code)


# Starlette sub-app mounted alongside MCP.
ingest_app = Starlette(
    routes=[
        Route("/ingest", handle_ingest, methods=["POST"]),
        Route("/healthz", handle_health, methods=["GET"]),
    ],
)
