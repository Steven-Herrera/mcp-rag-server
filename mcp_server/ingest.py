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
    _resolve_source: Resolves the source label for a file
    _ingest_pdf_streaming: Ingests a PDF in page batches to avoid OOM
    _flush_batch: Chunks and upserts a batch of pages
    _ingest_non_pdf: Ingests a non-PDF file
    ingest_path: Embeds and stores documents
    main: Entrypoint for document ingestion
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pymupdf
from loguru import logger

from mcp_server.chunking import chunk_text
from mcp_server.parsers import parse_file
from mcp_server.vectorstore import VectorStore


def _discover_files(root: Path) -> list[Path]:
    """Recursively discover ingestible files under root.

    Skips hidden directories.

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


def _resolve_source(file_path: Path, base: Path, source_prefix: str | None) -> str:
    """Resolve the source label for a file.

    Args:
        file_path: Path to the file.
        base: Base directory used for relative path resolution.
        source_prefix: Optional prefix to prepend.

    Returns:
        source: Source label string.
    """
    if not source_prefix:
        return str(file_path)
    try:
        relative = file_path.relative_to(base)
    except ValueError:
        relative = file_path
    return f"{source_prefix}/{relative}"


def _flush_batch(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    batch: list[str],
    store: VectorStore,
    source: str,
    metadata: dict[str, str],
    page_end: int,
    page_count: int,
) -> int:
    """Chunk and upsert a batch of page texts.

    Args:
        batch: List of page text strings.
        store: Vector store to upsert into.
        source: Source label for chunks.
        metadata: Base metadata dict.
        page_end: Index of the last page in the batch (0-based).
        page_count: Total pages in the document.

    Returns:
        count: Number of chunks ingested.
    """
    batch_text = "\n\n".join(batch)
    batch_meta = {**metadata, "pages": str(page_count)}
    chunks = chunk_text(batch_text, source=source, metadata=batch_meta)
    count = store.ingest_chunks(chunks)
    page_start = page_end - len(batch) + 2
    logger.info(
        "Ingested {} chunks from {} (pages {}-{})",
        count,
        source,
        page_start,
        page_end + 1,
    )
    return count


def _ingest_pdf_streaming(
    file_path: Path,
    store: VectorStore,
    source: str,
    metadata: dict[str, str],
    *,
    batch_size: int = 50,
) -> int:
    """Ingest a PDF file in page batches to avoid OOM on large files.

    Args:
        file_path: Path to the PDF file.
        store: Vector store to upsert into.
        source: Source label for chunks.
        metadata: Base metadata dict.
        batch_size: Number of pages to process per batch.

    Returns:
        total: Total chunks ingested.
    """
    doc = pymupdf.open(str(file_path))
    page_count = len(doc)
    total = 0
    batch: list[str] = []

    for i, page in enumerate(doc):
        text = page.get_text()
        page.reset_usage()
        if text.strip():
            batch.append(text)

        if (len(batch) >= batch_size or i == page_count - 1) and batch:
            total += _flush_batch(batch, store, source, metadata, i, page_count)
            batch.clear()

    doc.close()
    return total


def _ingest_non_pdf(
    file_path: Path,
    base: Path,
    store: VectorStore,
    source_prefix: str | None,
) -> int:
    """Ingest a single non-PDF file into the vector store.

    Args:
        file_path: Path to the file.
        base: Base directory for relative path resolution.
        store: Vector store to upsert into.
        source_prefix: Optional prefix to prepend to source paths.

    Returns:
        count: Number of chunks ingested, or 0 if skipped.
    """
    result = parse_file(file_path)
    if result is None:
        return 0

    text, metadata = result
    if not text.strip():
        logger.warning("Empty content: {}", file_path)
        return 0

    source = _resolve_source(file_path, base, source_prefix)
    metadata["path"] = str(file_path)
    if file_path.parent != Path("."):
        metadata["directory"] = str(file_path.parent)

    chunks = chunk_text(text, source=source, metadata=metadata)
    count = store.ingest_chunks(chunks)
    logger.info("Ingested {} chunks from {}", count, source)
    return count


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
        total: Number of chunks ingested.

    Raises:
        NotADirectoryError: Path doesn't exist.
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
        if file_path.suffix.lower() == ".pdf":
            source = _resolve_source(file_path, path, source_prefix)
            meta: dict[str, str] = {"path": str(file_path)}
            if file_path.parent != Path("."):
                meta["directory"] = str(file_path.parent)
            total += _ingest_pdf_streaming(file_path, store, source, meta)
        else:
            total += _ingest_non_pdf(file_path, path, store, source_prefix)

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
