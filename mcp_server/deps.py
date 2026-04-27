"""Shared VectorStore singleton accessor.

Both the MCP tool handlers and the HTTP upload endpoint need access
to the same VectorStore instance.  This module owns the singleton
so neither ``server.py`` nor ``upload.py`` duplicates the pattern.

Functions:
    _get_store: Returns the vector store
"""

from __future__ import annotations

from mcp_server.vectorstore import VectorStore

_store: VectorStore | None = None


def get_store() -> VectorStore:
    """Return the singleton VectorStore, creating it on first call.

    Returns:
        Initialised VectorStore connected to Qdrant.
    """
    global _store  # pylint: disable=global-statement
    if _store is None:
        _store = VectorStore()
        _store.ensure_collection()
    return _store
