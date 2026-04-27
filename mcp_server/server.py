"""MCP server exposing RAG search as a tool via Streamable HTTP transport.

This module defines the FastMCP server instance and registers the
search_documents tool that ChatGPT and Claude will call.

Functions:
    _get_store: Returns the vector store
    search_documents: Semantic similarity search
    list_sources: List collections and vector store status
"""

from __future__ import annotations

from loguru import logger
from mcp.server.fastmcp import FastMCP

from mcp_server.settings import settings
from mcp_server.vectorstore import VectorStore

mcp = FastMCP(settings.app_name, json_response=True)
_store: VectorStore | None = None


def _get_store() -> VectorStore:
    """Return the singleton VectorStore, creating it on first call.

    Returns:
        Initialised VectorStore connected to Qdrant.
    """
    global _store  # noqa: PLW0603
    if _store is None:
        _store = VectorStore()
        _store.ensure_collection()
    return _store


@mcp.tool()
def search_documents(
    query: str,
    top_k: int = 15,
    source_filter: str | None = None,
) -> list[dict]:
    """Search ingested documents using semantic similarity.

    Use this tool to find relevant document chunks based on a natural
    language query.  Returns the most similar text passages along with
    their source file paths and relevance scores.

    Args:
        query: Natural language search query describing what you're
            looking for.
        top_k: Maximum number of results to return (default 15).
        source_filter: Optional filter to restrict results to documents
            whose source path contains this substring.

    Returns:
        results: List of matching document chunks with text, source, score,
        and metadata.
    """
    logger.info("search_documents: query='{}' top_k={} filter={}", query, top_k, source_filter)
    store = _get_store()
    _results = store.search(query, top_k=top_k, source_filter=source_filter)
    logger.info("search_documents: returned {} results", len(_results))
    results = [r.model_dump() for r in _results]
    return results


@mcp.tool()
def list_sources() -> dict:
    """List collection statistics and status.

    Use this tool to check how many documents have been ingested
    and whether the vector store is healthy.

    Returns:
        Collection info with name, point count, and status.
    """
    store = _get_store()
    return store.collection_info().model_dump()
