"""CLI document ingestion tool.

Reads files (PDFs, text, code) from disk, chunks them, generates
embeddings, and upserts into the Qdrant collection.
Supports both individual files and recursive directory scanning.

Usage:

    mcp-ingest /path/to/document.pdf
    mcp-ingest /path/to/repo/
    mcp-ingest /path/to/repo/ --source-prefix "my-project"

Functions:
    _discover_files: Filters out irrelevant files from directory
    ingest_path: Embeds and stores documents
    main: Entrypoint for document ingestion
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from mcp_server.chunking import chunk_text
from mcp_server.parsers import parse_file
from mcp_server.vectorstore import VectorStore


def _discover_files(root: Path) -> list[Path]:
    """Recursively discover ingestible files under root

    Skips hidden directories, (e.g. __pycache__, .git, node_modules, .venv)

    Args:
        root: Root directory to scan.

    Returns:
        files: Sorted list of file paths.
    """
    skip_dirs = {".git", "__pycache__", "node_modules", ".venv", ".tox", ".ruff_cache"}
    files: list[Path] = []

    for item in sorted(root.rglob("*")):
        if any(part in skip_dirs for part in item.parts):
            continue
        if item.is_file():
            files.append(item)

    return files


def ingest_path(
    path: Path,
    store: VectorStore,
    *,
    source_prefix: str | None = None,
) -> int:
    """Ingest a file or directory into the vector store.

    Args:
        path: File or directory to ingest.
        store: Vector store to upsert into.
        source_prefix: Optional prefix to prepend to source paths.

    Returns:
        total: Number of chunks ingested

    Raises:
        NotADirectoryError: Path doesn't exist
    """
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = _discover_files(path)
    else:
        logger.error("Path does not exist: {}", path)
        raise NotADirectoryError(f"{path} does not exist")

    total = 0
    for file_path in files:
        result = parse_file(file_path)
        if result is None:
            continue

        text, metadata = result
        if not text.strip():
            logger.warning("Empty content: {}", file_path)
            continue

        source = str(file_path)
        if source_prefix:
            try:
                relative = file_path.relative_to(path)
            except ValueError:
                relative = file_path
            source = f"{source_prefix}/{relative}"

        metadata["path"] = str(file_path)
        if file_path.parent != Path("."):
            metadata["directory"] = str(file_path.parent)

        chunks = chunk_text(text, source=source, metadata=metadata)
        count = store.ingest_chunks(chunks)
        total += count
        logger.info("Ingested {} chunks from {}", count, source)

    return total


def main() -> None:
    """CLI entry point for document ingestion."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into the MCP RAG vector store.",
    )
    parser.add_argument(
        "path",
        type=Path,
        help="File or directory to ingest.",
    )
    parser.add_argument(
        "--source-prefix",
        type=str,
        default=None,
        help="Prefix for source paths (e.g. repo name).",
    )
    parser.add_argument(
        "--qdrant-url",
        type=str,
        default=None,
        help="Qdrant URL (overrides MCP_RAG_QDRANT_URL env var).",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Qdrant collection name (overrides MCP_RAG_COLLECTION_NAME env var).",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, level="INFO", format="{level: <8} | {message}")

    store = VectorStore(url=args.qdrant_url, collection_name=args.collection)
    store.ensure_collection()

    total = ingest_path(args.path, store, source_prefix=args.source_prefix)
    logger.info("Ingestion complete: {} total chunks", total)


if __name__ == "__main__":
    main()
