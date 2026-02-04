"""Interface contracts for bot_knows.

This module exports all Protocol-based interfaces for dependency injection.
"""

from bot_knows.interfaces.embedding import EmbeddingServiceInterface
from bot_knows.interfaces.graph import GraphServiceInterface
from bot_knows.interfaces.llm import LLMInterface
from bot_knows.interfaces.recall import RecallServiceInterface
from bot_knows.interfaces.storage import StorageInterface

__all__ = [
    "EmbeddingServiceInterface",
    "GraphServiceInterface",
    "StorageInterface",
    "RecallServiceInterface",
    "LLMInterface",
]
