"""Document parsers for supported file types.

Each parser extracts plain text and metadata from a specific format.
The parse_file dispatcher selects the right parser by extension.

Functions:
    parse_pdf: Extracts PDF text
    parse_text: Extract plain-text text
    parse_code: Extracts code text
    parse_file: Use a parser for a given file type
"""

from __future__ import annotations

from pathlib import Path

import pymupdf
from loguru import logger


def parse_pdf(path: Path) -> tuple[str, dict[str, str]]:
    """Extract text from a PDF using PyMuPDF.

    Args:
        path: Path to the PDF file.

    Returns:
        payload: Tuple of (full_text, metadata)
    """
    doc = pymupdf.open(str(path))
    page_count = len(doc)
    pages: list[str] = []
    for page in doc:
        text = page.get_text()
        if text.strip():
            pages.append(text)
        page.reset_usage()
    doc.close()

    metadata = {"type": "pdf", "pages": str(page_count)}
    return "\n\n".join(pages), metadata


def parse_text(path: Path) -> tuple[str, dict[str, str]]:
    """Read a plain UTF-8 text file.

    Args:
        path: Path to the text file.

    Returns:
        payload: Tuple of (full_text, metadata)
    """
    text = path.read_text(encoding="utf-8")
    metadata = {"type": "text"}
    payload = text, metadata
    return payload


def parse_code(path: Path) -> tuple[str, dict[str, str]]:
    """Read a source-code file, preserving path metadata.

    Args:
        path: Path to the code file.

    Returns:
        payload: Tuple of (full_text, metadata)
    """
    text = path.read_text(encoding="utf-8")
    metadata = {
        "type": "code",
        "language": path.suffix.lstrip("."),
        "filename": path.name,
    }
    payload = text, metadata
    return payload


_PARSERS: dict[str, callable] = {
    ".pdf": parse_pdf,
    ".txt": parse_text,
    ".md": parse_text,
    ".rst": parse_text,
    ".py": parse_code,
    ".js": parse_code,
    ".ts": parse_code,
    ".go": parse_code,
    ".rs": parse_code,
    ".java": parse_code,
    ".c": parse_code,
    ".cpp": parse_code,
    ".h": parse_code,
    ".hpp": parse_code,
    ".yaml": parse_code,
    ".yml": parse_code,
    ".json": parse_code,
    ".toml": parse_code,
    ".sh": parse_code,
    ".bash": parse_code,
    ".dockerfile": parse_code,
    ".makefile": parse_code,
}


def parse_file(path: Path) -> tuple[str, dict[str, str]] | None:
    """Parse a file into text and metadata using the appropriate parser.

    Args:
        path: Path to the file.

    Returns:
        payload: Tuple of (text, metadata)

    Raises:
        TypeError: Unsupported file type
    """
    suffix = path.suffix.lower()
    if path.name.lower() in {"dockerfile", "makefile"}:
        suffix = f".{path.name.lower()}"

    parser = _PARSERS.get(suffix)
    if parser is None:
        logger.warning("Unsupported file type: {}", path)
        raise TypeError(f"Unsupported file type: {path}")

    logger.debug("Parsing {} with {} parser", path, suffix)
    payload = parser(path)
    return payload
