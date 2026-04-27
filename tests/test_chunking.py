"""Tests for chunking and parser utilities.

Classes:
    TestChunkText: Tests chunking strategies
    TestParsers: Tests parsing strategies
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mcp_server.chunking import chunk_text
from mcp_server.parsers import parse_file


class TestChunkText:
    """Text chunking tests."""

    def test_empty_text(self) -> None:
        """Empty text produces no chunks."""
        assert chunk_text("", "test.txt") == []
        assert chunk_text("   ", "test.txt") == []

    def test_short_text_single_chunk(self) -> None:
        """Text shorter than chunk_size produces one chunk."""
        chunks = chunk_text("Hello world", "test.txt", chunk_size=100, chunk_overlap=10)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world"
        assert chunks[0].source == "test.txt"
        assert chunks[0].chunk_index == 0

    def test_long_text_multiple_chunks(self) -> None:
        """Long text is split into multiple overlapping chunks."""
        text = "word " * 500  # 2500 chars
        chunks = chunk_text(text, "long.txt", chunk_size=200, chunk_overlap=50)
        assert len(chunks) > 1
        # Verify ordering.
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.source == "long.txt"

    def test_metadata_preserved(self) -> None:
        """Custom metadata is attached to every chunk."""
        chunks = chunk_text(
            "Some text content here",
            "doc.pdf",
            chunk_size=100,
            chunk_overlap=10,
            metadata={"type": "pdf", "pages": "3"},
        )
        assert len(chunks) == 1
        assert chunks[0].metadata["type"] == "pdf"
        assert chunks[0].metadata["pages"] == "3"

    def test_paragraph_boundary(self) -> None:
        """Chunker prefers paragraph boundaries when possible."""
        para1 = "A" * 80
        para2 = "B" * 80
        text = f"{para1}\n\n{para2}"
        chunks = chunk_text(text, "test.txt", chunk_size=100, chunk_overlap=20)
        # First chunk should contain para1, not split mid-word.
        assert chunks[0].text.startswith("A")


class TestParsers:
    """Parser dispatch tests."""

    def test_parse_text_file(self, tmp_path: Path) -> None:
        """Plain text parser reads UTF-8 content."""

        f = tmp_path / "test.txt"
        f.write_text("Hello world", encoding="utf-8")
        result = parse_file(f)
        assert result is not None
        text, meta = result
        assert text == "Hello world"
        assert meta["type"] == "text"

    def test_parse_python_file(self, tmp_path: Path) -> None:
        """Code parser reads .py files and sets language metadata."""

        f = tmp_path / "example.py"
        f.write_text("print('hello')", encoding="utf-8")
        result = parse_file(f)
        assert result is not None
        text, meta = result
        assert "print" in text
        assert meta["type"] == "code"
        assert meta["language"] == "py"

    def test_unsupported_extension(self, tmp_path: Path) -> None:
        """Unsupported file types return None."""

        f = tmp_path / "image.png"
        f.write_bytes(b"\x89PNG")
        with pytest.raises(TypeError):
            parse_file(f)

    def test_parse_markdown(self, tmp_path: Path) -> None:
        """Markdown files are parsed as text."""

        f = tmp_path / "readme.md"
        f.write_text("# Title\n\nContent here", encoding="utf-8")
        result = parse_file(f)
        assert result is not None
        text, meta = result
        assert "# Title" in text
        assert meta["type"] == "text"
