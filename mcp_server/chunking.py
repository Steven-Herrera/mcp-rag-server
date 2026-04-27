"""Text chunking utilities for document ingestion.

Splits text into overlapping chunks suitable for embedding and
retrieval.  Respects paragraph and sentence boundaries where possible.

Classes:
    Chunk: Represents a chunk of text

Functions:
    chunk_text: Paragraph or space-based chunking
"""

from __future__ import annotations

from pydantic import BaseModel

from mcp_server.settings import settings


class Chunk(BaseModel):
    """A single text chunk with source metadata.

    Attributes:
        text: The chunk content.
        source: Original file path or URI.
        chunk_index: Zero-based index within the source document.
        metadata: Arbitrary key-value metadata from the source.
    """

    text: str
    source: str
    chunk_index: int
    metadata: dict[str, str] = {}


def chunk_text(
    text: str,
    source: str,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    metadata: dict[str, str] | None = None,
) -> list[Chunk]:
    """Split text into overlapping chunks.

    Tries to break on paragraph boundaries (double newlines), then
    falls back to the nearest space before chunk_size.

    Args:
        text: Full document text.
        source: Source identifier (file path, URL, etc.).
        chunk_size: Max characters per chunk (default from settings).
        chunk_overlap: Overlap between consecutive chunks.
        metadata: Extra metadata attached to every chunk.

    Returns:
        chunks: Ordered list of Chunk objects.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap
    metadata = metadata or {}

    text = text.strip()
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    idx = 0

    while start < len(text):
        end = start + chunk_size

        if end < len(text):
            # Try paragraph boundary first.
            boundary = text.rfind("\n\n", start, end)
            if boundary == -1 or boundary <= start:
                # Fall back to nearest space.
                boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary + 1

        chunk_text_str = text[start:end].strip()
        if chunk_text_str:
            chunks.append(
                Chunk(
                    text=chunk_text_str,
                    source=source,
                    chunk_index=idx,
                    metadata=metadata,
                )
            )
            idx += 1

        start = max(start + 1, end - chunk_overlap)

    return chunks
