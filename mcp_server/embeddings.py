"""Embedding service wrapping a local sentence-transformers model.

The all-MiniLM-L6-v2 model is loaded lazily on first use and
cached for the process lifetime.

Functions:
    _load_model: Loads embedding model
    embed_texts: Batch encode texts
    embed_query: Encodes single queries
"""

from __future__ import annotations

from functools import lru_cache

from loguru import logger
from sentence_transformers import SentenceTransformer

from mcp_server.settings import settings


@lru_cache(maxsize=1)
def _load_model() -> SentenceTransformer:
    """Load and cache the sentence-transformers model.

    Returns:
        The loaded model instance.
    """
    logger.info("Loading embedding model: {}", settings.embedding_model)
    model = SentenceTransformer(settings.embedding_model, device="cpu")
    logger.info("Embedding model loaded (dim={})", settings.embedding_dimension)
    return model


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Encode a batch of texts into embedding vectors.

    Args:
        texts: List of text strings to embed.

    Returns:
        embeddings_lst: List of embedding vectors
    """
    model = _load_model()
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    embeddings_lst = embeddings.tolist()
    return embeddings_lst


def embed_query(query: str) -> list[float]:
    """Encode a single query string.

    Args:
        query: The search query.

    Returns:
        embedded_query: Embedding vector
    """
    embedded_query = embed_texts([query])[0]
    return embedded_query
