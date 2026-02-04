"""bot_knows - Framework-agnostic Python library for graph-backed personal knowledge bases.

This package provides tools for:
- Ingesting chats from multiple sources (ChatGPT, Claude, custom JSON)
- Classifying and organizing chat data
- Extracting semantic topics with deduplication
- Building a graph-backed knowledge base
- Evidence-weighted recall with spaced repetition

Example usage:
    from bot_knows import (
        BotKnowsConfig,
        ChatProcessingService,
        ImportAdapterRegistry,
        ChatIngest,
    )

    # Load configuration
    config = BotKnowsConfig()

    # Import a ChatGPT export
    adapter = ImportAdapterRegistry.create("chatgpt")
    chats = adapter.parse_file("conversations.json")

    # Process each chat
    for chat_ingest in chats:
        chat, is_new = await processor.process(chat_ingest)
"""

__version__ = "0.1.0"

# Configuration
from bot_knows.config import (
    BotKnowsConfig,
    LLMSettings,
    MongoSettings,
    Neo4jSettings,
    RedisSettings,
)

# Import adapters
from bot_knows.importers.base import ChatImportAdapter
from bot_knows.importers.registry import ImportAdapterRegistry

# Interfaces
from bot_knows.interfaces.embedding import EmbeddingServiceInterface
from bot_knows.interfaces.graph import GraphServiceInterface
from bot_knows.interfaces.llm import LLMInterface
from bot_knows.interfaces.recall import RecallServiceInterface
from bot_knows.interfaces.storage import StorageInterface

# Public DTOs
from bot_knows.models.chat import ChatCategory, ChatDTO
from bot_knows.models.ingest import ChatIngest, IngestMessage
from bot_knows.models.message import MessageDTO
from bot_knows.models.recall import RecallItemDTO, TopicRecallStateDTO
from bot_knows.models.topic import TopicDTO, TopicEvidenceDTO

# Services
from bot_knows.services.chat_processing import ChatProcessingService
from bot_knows.services.recall_service import RecallService

__all__ = [
    # Config
    "BotKnowsConfig",
    "ChatCategory",
    "ChatDTO",
    # Importers
    "ChatImportAdapter",
    # Models
    "ChatIngest",
    # Services
    "ChatProcessingService",
    # Interfaces
    "EmbeddingServiceInterface",
    "GraphServiceInterface",
    "ImportAdapterRegistry",
    "IngestMessage",
    "LLMInterface",
    "LLMSettings",
    "MessageDTO",
    "MongoSettings",
    "Neo4jSettings",
    "RecallItemDTO",
    "RecallService",
    "RecallServiceInterface",
    "RedisSettings",
    "StorageInterface",
    "TopicDTO",
    "TopicEvidenceDTO",
    "TopicRecallStateDTO",
    # Version
    "__version__",
]
