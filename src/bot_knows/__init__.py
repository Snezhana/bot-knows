"""bot_knows - Framework-agnostic Python library for graph-backed personal knowledge bases.

This package provides tools for:
- Ingesting chats from multiple sources (ChatGPT, Claude, custom JSON)
- Classifying and organizing chat data
- Extracting semantic topics with deduplication
- Building a graph-backed knowledge base
- Evidence-weighted recall with spaced repetition

Example usage:
    from bot_knows import (
        BotKnows,
        MongoStorageRepository,
        Neo4jGraphRepository,
        OpenAIProvider,
        ChatGPTAdapter,
    )

    # Simple usage - config loaded from .env automatically
    async with BotKnows(
        storage_class=MongoStorageRepository,
        graphdb_class=Neo4jGraphRepository,
        llm_class=OpenAIProvider,
    ) as bk:
        result = await bk.insert_chats("conversations.json", ChatGPTAdapter)
        topics = await bk.get_chat_topics(chat_id)
"""

__version__ = "0.1.0"

# Orchestrator
# Interfaces
from bot_knows.importers.base import ChatImportAdapter

# Import adapters
from bot_knows.importers.chatgpt import ChatGPTAdapter
from bot_knows.importers.claude import ClaudeAdapter
from bot_knows.importers.generic_json import GenericJSONAdapter
from bot_knows.infra.llm.anthropic_provider import AnthropicProvider
from bot_knows.infra.llm.openai_provider import OpenAIProvider

# Implementations
from bot_knows.infra.mongo.repositories import MongoStorageRepository
from bot_knows.infra.neo4j.graph_repository import Neo4jGraphRepository
from bot_knows.interfaces.embedding import EmbeddingServiceInterface
from bot_knows.interfaces.graph import GraphServiceInterface
from bot_knows.interfaces.llm import LLMInterface
from bot_knows.interfaces.storage import StorageInterface
from bot_knows.orchestrator import BotKnows, InsertResult

__all__ = [  # noqa: RUF022
    # Orchestrator
    "BotKnows",
    "InsertResult",
    # Implementations
    "MongoStorageRepository",
    "Neo4jGraphRepository",
    "OpenAIProvider",
    "AnthropicProvider",
    # Import adapters
    "ChatGPTAdapter",
    "ClaudeAdapter",
    "GenericJSONAdapter",
    # Interfaces
    "ChatImportAdapter",
    "EmbeddingServiceInterface",
    "GraphServiceInterface",
    "LLMInterface",
    "StorageInterface",
]
