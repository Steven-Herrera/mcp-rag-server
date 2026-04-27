"""MCP RAG server entry point.

Configures logging and starts the FastMCP server with
Streamable HTTP transport.  The HTTP upload/ingest endpoint is
mounted alongside the MCP endpoint on the same port.

Classes:
    _IntercepteHandler: Intercepts logging from other packages and re-routes them to loguru

Functions:
    _configure_loguru: Sets up loguru as the universal logger
    main: Starts the MCP server
"""

from __future__ import annotations

import logging
import sys

import uvicorn
from loguru import logger

from mcp_server.server import mcp
from mcp_server.settings import settings
from mcp_server.upload import ingest_app


class _InterceptHandler(logging.Handler):
    """Bridge stdlib logging into loguru."""

    def emit(self, record: logging.LogRecord) -> None:
        """Forward a stdlib log record to loguru.

        Args:
            record: The stdlib log record.
        """
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def _configure_logging(level: str = "INFO") -> None:
    """Set up loguru as the single logging sink.

    Args:
        level: Minimum log level.
    """
    logger.remove()
    logger.add(
        sys.stderr,
        level=level,
        format=(
            "<green>{time:YYYY-MM-DDTHH:mm:ss.SSSZ}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        backtrace=True,
        diagnose=False,
    )
    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "httpx", "sentence_transformers"):
        logging.getLogger(name).handlers = [_InterceptHandler()]


def main() -> None:
    """Start the MCP RAG server with Streamable HTTP transport.

    The MCP endpoint lives at /mcp and the ingest upload endpoint
    at /ingest.  Both are served on the same port.
    """

    _configure_logging(settings.log_level)
    logger.info(
        "Starting {} on {}:{}",
        settings.app_name,
        settings.host,
        settings.port,
    )

    app = mcp.streamable_http_app()
    app.mount("/api", ingest_app)

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
