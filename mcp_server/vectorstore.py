"""Qdrant vector store client.

Handles collection creation, document upsert, and similarity search.
Uses the Qdrant Python client's HTTP transport.

Classes:
    SearchResult: Single search result
    CollectionInfo: Collection stats
    VectorStore: RAG database instance
"""

from __future__ import annotations

import uuid

from loguru import logger
from pydantic import BaseModel
from qdrant_client import QdrantClient, models

from mcp_server.chunking import Chunk
from mcp_server.embeddings import embed_query, embed_texts
from mcp_server.settings import settings


class SearchResult(BaseModel):
    """A single search result returned by VectorStore.search

    Attributes:
        text: The matched chunk text.
        source: Source file path or identifier.
        score: Cosine similarity score (0.0-1.0).
        chunk_index: Position of this chunk within the source document.
        metadata: Any additional payload fields from the chunk.
    """

    text: str
    source: str
    score: float
    chunk_index: int = 0
    metadata: dict[str, str] = {}


class CollectionInfo(BaseModel):
    """Basic collection statistics from Qdrant.

    Attributes:
        name: Collection name.
        points_count: Number of stored vectors.
        status: Collection status string
    """

    name: str
    points_count: int
    status: str


# Keys stored directly on SearchResult fields (not in metadata).
_STRUCTURED_KEYS = frozenset({"text", "source", "chunk_index"})


class VectorStore:
    """Qdrant vector store for document chunks.

    Manages a single collection and provides high-level ingest and
    search operations.

    Args:
        url: Qdrant HTTP URL.
        api_key: Optional Qdrant API key.
        collection_name: Name of the target collection.
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        collection_name: str | None = None,
    ) -> None:
        self._url = url or settings.qdrant_url
        self._api_key = api_key or settings.qdrant_api_key
        self._collection = collection_name or settings.collection_name
        self._client = QdrantClient(url=self._url, api_key=self._api_key, timeout=120)

    def ensure_collection(self) -> None:
        """Create the collection if it does not already exist."""
        collections = [c.name for c in self._client.get_collections().collections]
        if self._collection in collections:
            logger.info("Collection '{}' already exists", self._collection)
            return

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=models.VectorParams(
                size=settings.embedding_dimension,
                distance=models.Distance.COSINE,
            ),
        )
        logger.info(
            "Created collection '{}' (dim={}, cosine)",
            self._collection,
            settings.embedding_dimension,
        )

    def ingest_chunks(self, chunks: list[Chunk]) -> int:
        """Embed and upsert a batch of chunks into Qdrant.

        Args:
            chunks: Document chunks to ingest.

        Returns:
            num_points_upserted: Number of points upserted.
        """
        if not chunks:
            return 0

        texts = [c.text for c in chunks]
        vectors = embed_texts(texts)

        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{chunk.source}:{chunk.chunk_index}")),
                vector=vec,
                payload={
                    "text": chunk.text,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                    **chunk.metadata,
                },
            )
            for chunk, vec in zip(chunks, vectors, strict=True)
        ]

        self._client.upsert(
            collection_name=self._collection,
            points=points,
        )
        logger.info("Upserted {} chunks from '{}'", len(points), chunks[0].source)
        num_points_upserted = len(points)
        return num_points_upserted

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        source_filter: str | None = None,
    ) -> list[SearchResult]:
        """Search for chunks similar to query

        Args:
            query: Natural language search query.
            top_k: Number of results to return.
            source_filter: Optional source path prefix filter.

        Returns:
            search_results: Ordered list of SearchResult objects, highest score first.
        """
        top_k = top_k or settings.top_k
        query_vector = embed_query(query)

        query_filter = None
        if source_filter:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="source",
                        match=models.MatchText(text=source_filter),
                    )
                ]
            )

        results = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )

        search_results: list[SearchResult] = []
        for point in results.points:
            payload = point.payload or {}
            extra_metadata = {k: str(v) for k, v in payload.items() if k not in _STRUCTURED_KEYS}
            search_results.append(
                SearchResult(
                    text=payload.get("text", ""),
                    source=payload.get("source", ""),
                    score=point.score,
                    chunk_index=payload.get("chunk_index", 0),
                    metadata=extra_metadata,
                )
            )
        return search_results

    def collection_info(self) -> CollectionInfo:
        """Return basic collection statistics.

        Returns:
            CollectionInfo with name, point count, and status.
        """
        info = self._client.get_collection(self._collection)
        return CollectionInfo(
            name=self._collection,
            points_count=info.points_count or 0,
            status=info.status.value,
        )

    def delete_by_source(self, source: str) -> int:
        """Delete all points matching a source prefix.

        Qdrant's delete operation doesn't return a count of deleted points.
        So this method reads points_count before and after the delete and
        takes the difference. The count is approximate because Qdrant may
        not have fully compacted the storage by the time the second get_collection
        call runs i.e. there's a small race window.
        There's no way to get an exact count from Qdrant's API for a filter-based delete.
        Exact counts would require querying the matching points first, counting them,
        then deleting by IDs, but that's slower and not worth it for a logging message.

        Args:
            source: Source path or prefix to match.

        Returns:
            Approximate number of deleted points.
        """
        info_before = self._client.get_collection(self._collection)
        self._client.delete(
            collection_name=self._collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="source",
                            match=models.MatchText(text=source),
                        )
                    ]
                )
            ),
        )
        info_after = self._client.get_collection(self._collection)
        deleted = (info_before.points_count or 0) - (info_after.points_count or 0)
        logger.info("Deleted ~{} points matching source='{}'", deleted, source)
        return deleted
