"""Service layer for bot_knows.

This module exports the main service entry points.
"""

from bot_knows.services.chat_processing import ChatProcessingService
from bot_knows.services.dedup_service import DedupAction, DedupResult, DedupService
from bot_knows.services.graph_service import GraphService
from bot_knows.services.message_builder import MessageBuilder
from bot_knows.services.recall_service import CONTEXT_WEIGHTS, RecallService
from bot_knows.services.topic_extraction import TopicCandidate, TopicExtractionService

__all__ = [
    "CONTEXT_WEIGHTS",
    "ChatProcessingService",
    "DedupAction",
    "DedupResult",
    "DedupService",
    "GraphService",
    "MessageBuilder",
    "RecallService",
    "TopicCandidate",
    "TopicExtractionService",
]
