"""Unit tests for bot_knows models."""

import pytest
from pydantic import ValidationError

from bot_knows.models.chat import ChatCategory, ChatDTO
from bot_knows.models.ingest import ChatIngest, IngestMessage
from bot_knows.models.message import MessageDTO
from bot_knows.models.recall import TopicRecallStateDTO
from bot_knows.models.topic import TopicDTO, TopicEvidenceDTO


class TestIngestMessage:
    """Tests for IngestMessage model."""

    def test_valid_user_message(self) -> None:
        msg = IngestMessage(
            role="user",
            content="Hello",
            timestamp=1704067200,
            chat_id="chat1",
        )
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.timestamp == 1704067200
        assert msg.schema_version == 1

    def test_valid_assistant_message(self) -> None:
        msg = IngestMessage(
            role="assistant",
            content="Hi there!",
            timestamp=1704067200,
            chat_id="chat1",
        )
        assert msg.role == "assistant"

    def test_valid_system_message(self) -> None:
        msg = IngestMessage(
            role="system",
            content="You are a helpful assistant.",
            timestamp=1704067200,
            chat_id="chat1",
        )
        assert msg.role == "system"

    def test_invalid_role(self) -> None:
        with pytest.raises(ValidationError):
            IngestMessage(
                role="invalid",  # type: ignore[arg-type]
                content="Hello",
                timestamp=1704067200,
                chat_id="chat1",
            )

    def test_frozen_model(self) -> None:
        msg = IngestMessage(
            role="user",
            content="Hello",
            timestamp=1704067200,
            chat_id="chat1",
        )
        with pytest.raises(ValidationError):
            msg.content = "Changed"  # type: ignore[misc]


class TestChatIngest:
    """Tests for ChatIngest model."""

    def test_valid_chat_ingest(self) -> None:
        messages = [
            IngestMessage(role="user", content="Hi", timestamp=1704067200, chat_id="c1"),
        ]
        chat = ChatIngest(
            source="test",
            imported_chat_timestamp=1704067200,
            title="Test Chat",
            messages=messages,
        )
        assert chat.source == "test"
        assert chat.title == "Test Chat"
        assert chat.message_count == 1
        assert chat.has_messages is True

    def test_empty_messages(self) -> None:
        chat = ChatIngest(
            source="test",
            imported_chat_timestamp=1704067200,
            title=None,
            messages=[],
        )
        assert chat.message_count == 0
        assert chat.has_messages is False

    def test_frozen_model(self) -> None:
        chat = ChatIngest(
            source="test",
            imported_chat_timestamp=1704067200,
            title="Test",
            messages=[],
        )
        with pytest.raises(ValidationError):
            chat.title = "Changed"  # type: ignore[misc]


class TestChatDTO:
    """Tests for ChatDTO model."""

    def test_valid_chat(self) -> None:
        chat = ChatDTO(
            id="abc123",
            title="Test Chat",
            source="chatgpt",
            category=ChatCategory.CODING,
            tags=["python"],
            created_on=1704067200,
        )
        assert chat.id == "abc123"
        assert chat.category == ChatCategory.CODING

    def test_default_category(self) -> None:
        chat = ChatDTO(
            id="abc123",
            title="Test",
            source="test",
            created_on=1704067200,
        )
        assert chat.category == ChatCategory.GENERAL
        assert chat.tags == []


class TestMessageDTO:
    """Tests for MessageDTO model."""

    def test_valid_message(self) -> None:
        msg = MessageDTO(
            message_id="msg1",
            chat_id="chat1",
            user_content="Hello",
            assistant_content="Hi there!",
            created_on=1704067200,
        )
        assert msg.combined_content == "User: Hello\n\nAssistant: Hi there!"
        assert msg.is_empty is False

    def test_empty_message(self) -> None:
        msg = MessageDTO(
            message_id="msg1",
            chat_id="chat1",
            created_on=1704067200,
        )
        assert msg.is_empty is True
        assert msg.combined_content == ""


class TestTopicDTO:
    """Tests for TopicDTO model."""

    def test_valid_topic(self) -> None:
        topic = TopicDTO(
            topic_id="topic1",
            canonical_name="Python",
            centroid_embedding=[0.1, 0.2, 0.3],
            evidence_count=1,
            importance=0.5,
            recall_strength=0.3,
        )
        assert topic.canonical_name == "Python"
        assert len(topic.centroid_embedding) == 3

    def test_with_updated_centroid(self) -> None:
        topic = TopicDTO(
            topic_id="topic1",
            canonical_name="Python",
            centroid_embedding=[1.0, 0.0],
            evidence_count=1,
            importance=0.5,
            recall_strength=0.3,
        )
        updated = topic.with_updated_centroid([0.0, 1.0])
        assert updated.evidence_count == 2
        assert updated.centroid_embedding == [0.5, 0.5]  # (1+0)/2, (0+1)/2

    def test_first_centroid_update(self) -> None:
        topic = TopicDTO(
            topic_id="topic1",
            canonical_name="Python",
            centroid_embedding=[],
            evidence_count=0,
        )
        updated = topic.with_updated_centroid([1.0, 2.0])
        assert updated.centroid_embedding == [1.0, 2.0]
        assert updated.evidence_count == 1

    def test_importance_bounds(self) -> None:
        with pytest.raises(ValidationError):
            TopicDTO(
                topic_id="t1",
                canonical_name="Test",
                importance=1.5,  # > 1.0
            )

        with pytest.raises(ValidationError):
            TopicDTO(
                topic_id="t1",
                canonical_name="Test",
                importance=-0.1,  # < 0.0
            )


class TestTopicEvidenceDTO:
    """Tests for TopicEvidenceDTO model."""

    def test_valid_evidence(self) -> None:
        evidence = TopicEvidenceDTO(
            evidence_id="ev1",
            topic_id="topic1",
            extracted_name="python async",
            source_message_id="msg1",
            confidence=0.9,
            timestamp=1704067200,
        )
        assert evidence.confidence == 0.9

    def test_confidence_bounds(self) -> None:
        with pytest.raises(ValidationError):
            TopicEvidenceDTO(
                evidence_id="ev1",
                topic_id="t1",
                extracted_name="test",
                source_message_id="m1",
                confidence=1.5,  # > 1.0
                timestamp=0,
            )


class TestTopicRecallStateDTO:
    """Tests for TopicRecallStateDTO model."""

    def test_valid_state(self) -> None:
        state = TopicRecallStateDTO(
            topic_id="topic1",
            strength=0.5,
            last_seen=1704067200,
            last_updated=1704067200,
            stability=1.5,
        )
        assert state.strength == 0.5
        assert state.stability == 1.5

    def test_strength_bounds(self) -> None:
        with pytest.raises(ValidationError):
            TopicRecallStateDTO(
                topic_id="t1",
                strength=1.5,  # > 1.0
                last_seen=0,
                last_updated=0,
            )
