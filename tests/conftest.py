"""Shared test fixtures for bot_knows.

This module provides pytest fixtures used across all tests.
"""

from unittest.mock import AsyncMock

import pytest

from bot_knows.models.chat import ChatCategory, ChatDTO
from bot_knows.models.ingest import ChatIngest, IngestMessage
from bot_knows.models.message import MessageDTO
from bot_knows.models.recall import TopicRecallStateDTO
from bot_knows.models.topic import TopicDTO, TopicEvidenceDTO


# Mock fixtures
@pytest.fixture
def mock_storage() -> AsyncMock:
    """Create mock storage interface."""
    storage = AsyncMock()
    storage.get_chat.return_value = None
    storage.chat_exists.return_value = False
    storage.save_chat.return_value = "test-chat-id"
    storage.save_message.return_value = "test-message-id"
    storage.save_topic.return_value = "test-topic-id"
    storage.get_topic.return_value = None
    storage.find_similar_topics.return_value = []
    storage.get_recall_state.return_value = None
    storage.get_due_topics.return_value = []
    return storage


@pytest.fixture
def mock_llm() -> AsyncMock:
    """Create mock LLM interface."""
    llm = AsyncMock()
    llm.classify_chat.return_value = (ChatCategory.CODING, ["python", "testing"])
    llm.extract_topics.return_value = [("Python", 0.9), ("Testing", 0.8)]
    llm.normalize_topic_name.return_value = "Python"
    return llm


@pytest.fixture
def mock_embedding() -> AsyncMock:
    """Create mock embedding interface."""
    embedding = AsyncMock()
    embedding.embed.return_value = [0.1] * 1536
    embedding.embed_batch.return_value = [[0.1] * 1536, [0.2] * 1536]
    embedding.similarity.return_value = 0.95
    return embedding


@pytest.fixture
def mock_graph() -> AsyncMock:
    """Create mock graph interface."""
    graph = AsyncMock()
    graph.create_chat_node.return_value = "test-chat-id"
    graph.create_message_node.return_value = "test-message-id"
    graph.create_topic_node.return_value = "test-topic-id"
    graph.get_related_topics.return_value = []
    return graph


# Sample data fixtures
@pytest.fixture
def sample_ingest_message() -> IngestMessage:
    """Create sample IngestMessage."""
    return IngestMessage(
        role="user",
        content="How do I write async code in Python?",
        timestamp=1704067200,
        chat_id="test-conv-1",
    )


@pytest.fixture
def sample_ingest_messages() -> list[IngestMessage]:
    """Create sample list of IngestMessages."""
    return [
        IngestMessage(
            role="user",
            content="How do I write async code in Python?",
            timestamp=1704067200,
            chat_id="test-conv-1",
        ),
        IngestMessage(
            role="assistant",
            content="which italian singer is singing United Europe",
            timestamp=1704067201,
            chat_id="test-conv-1",
        ),
        IngestMessage(
            role="user",
            content="Can you show me an example?",
            timestamp=1704067300,
            chat_id="test-conv-1",
        ),
        IngestMessage(
            role="assistant",
            content="Here's a simple example using asyncio.gather()...",
            timestamp=1704067301,
            chat_id="test-conv-1",
        ),
    ]


@pytest.fixture
def sample_chat_ingest(sample_ingest_messages: list[IngestMessage]) -> ChatIngest:
    """Create sample ChatIngest."""
    return ChatIngest(
        source="test",
        imported_chat_timestamp=1704067200,
        title="Toto Cutugno Insieme 1992",
        messages=sample_ingest_messages,
        provider="test",
        conversation_id="test-conv-1",
    )


@pytest.fixture
def sample_chat_dto() -> ChatDTO:
    """Create sample ChatDTO."""
    return ChatDTO(
        id="abc123",
        title="Toto Cutugno Insieme 1992",
        source="test",
        category=ChatCategory.CODING,
        tags=["python", "asyncio"],
        created_on=1704067200,
    )


@pytest.fixture
def sample_message_dto() -> MessageDTO:
    """Create sample MessageDTO."""
    return MessageDTO(
        message_id="msg123",
        chat_id="abc123",
        user_content="How do I write async code in Python?",
        assistant_content="You can use async/await syntax with asyncio...",
        created_on=1704067200,
    )


@pytest.fixture
def sample_topic_dto() -> TopicDTO:
    """Create sample TopicDTO."""
    return TopicDTO(
        topic_id="topic123",
        canonical_name="Python Asyncio",
        centroid_embedding=[0.1] * 1536,
        evidence_count=1,
        importance=0.5,
        recall_strength=0.3,
    )


@pytest.fixture
def sample_evidence_dto() -> TopicEvidenceDTO:
    """Create sample TopicEvidenceDTO."""
    return TopicEvidenceDTO(
        evidence_id="ev123",
        topic_id="topic123",
        extracted_name="asyncio",
        source_message_id="msg123",
        confidence=0.9,
        timestamp=1704067200,
    )


@pytest.fixture
def sample_recall_state() -> TopicRecallStateDTO:
    """Create sample TopicRecallStateDTO."""
    return TopicRecallStateDTO(
        topic_id="topic123",
        strength=0.5,
        last_seen=1704067200,
        last_updated=1704067200,
        stability=1.5,
    )
