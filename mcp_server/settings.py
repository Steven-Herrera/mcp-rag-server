"""Runtime configuration for the MCP RAG server.

All settings are resolved from environment variables prefixed with MCP_RAG_
which are in mcp-server.yaml

Classes:
    Settings: MCP RAG Config
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """MCP RAG server configuration.

    Attributes:
        app_name: Human-readable service name.
        host: Bind address for the MCP HTTP transport.
        port: Bind port for the MCP HTTP transport.
        mcp_path: URL path for the MCP Streamable HTTP endpoint.
        qdrant_url: Qdrant gRPC/HTTP URL.
        qdrant_api_key: Optional Qdrant API key.
        collection_name: Qdrant collection for document chunks.
        embedding_model: HuggingFace sentence-transformers model name.
        embedding_dimension: Dimensionality of the embedding vectors.
        chunk_size: Target chunk size in characters for text splitting.
        chunk_overlap: Overlap between adjacent chunks in characters.
        top_k: Default number of results for similarity search.
        log_level: Minimum log level for loguru.
    """

    model_config = SettingsConfigDict(env_prefix="MCP_RAG_", case_sensitive=False)

    app_name: str = "mcp-rag-server"
    host: str = "0.0.0.0"
    port: int = 9000
    mcp_path: str = "/mcp"

    qdrant_url: str = "http://qdrant.mcp.svc.cluster.local:6333"
    qdrant_api_key: str | None = None
    collection_name: str = "documents"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384

    chunk_size: int = 1000
    chunk_overlap: int = 200

    top_k: int = 5

    log_level: str = "INFO"


settings = Settings()
