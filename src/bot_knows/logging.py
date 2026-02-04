"""Structured logging for bot_knows.

This module provides a configured structlog logger with JSON output
for production and pretty console output for development.
"""

import logging
import sys
from typing import Any

import structlog

__all__ = [
    "configure_logging",
    "get_logger",
]


def configure_logging(
    level: int = logging.INFO,
    json_output: bool = False,
    add_timestamp: bool = True,
) -> None:
    """Configure structlog for the application.

    Args:
        level: Logging level (default: INFO)
        json_output: If True, output JSON; if False, pretty console output
        add_timestamp: If True, add ISO timestamp to log entries
    """
    # Common processors
    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if add_timestamp:
        processors.insert(0, structlog.processors.TimeStamper(fmt="iso"))

    if json_output:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.plain_traceback,
            )
        )

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    # Set levels for noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("motor").setLevel(logging.WARNING)
    logging.getLogger("neo4j").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a configured structlog logger.

    Args:
        name: Logger name (usually __name__ of the calling module)

    Returns:
        Configured structlog BoundLogger instance
    """
    return structlog.get_logger(name)


# Convenience: configure with defaults on import if not already configured
_configured = False


def _ensure_configured() -> None:
    """Ensure logging is configured with defaults."""
    global _configured
    if not _configured:
        configure_logging()
        _configured = True


_ensure_configured()
