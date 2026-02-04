"""Unit tests for bot_knows services."""

import pytest
from unittest.mock import AsyncMock

from bot_knows.models.chat import ChatCategory, ChatDTO
from bot_knows.models.ingest import ChatIngest, IngestMessage
from bot_knows.models.topic import TopicDTO
from bot_knows.services.chat_processing import ChatProcessingService
from bot_knows.services.dedup_service import DedupAction, DedupService
from bot_knows.services.message_builder import MessageBuilder


class TestChatProcessingService:
    """Tests for ChatProcessingService."""

    @pytest.mark.asyncio
    async def test_process_new_chat(
        self,
        mock_storage: AsyncMock,
        mock_llm: AsyncMock,
        sample_chat_ingest: ChatIngest,
    ) -> None:
        service = ChatProcessingService(mock_storage, mock_llm)

        chat, is_new = await service.process(sample_chat_ingest)

        assert is_new is True
        assert chat.title == sample_chat_ingest.title
        assert chat.source == sample_chat_ingest.source
        assert chat.category == ChatCategory.CODING
        mock_storage.save_chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_existing_chat(
        self,
        mock_storage: AsyncMock,
        mock_llm: AsyncMock,
        sample_chat_ingest: ChatIngest,
        sample_chat_dto: ChatDTO,
    ) -> None:
        # Simulate existing chat
        mock_storage.get_chat.return_value = sample_chat_dto

        service = ChatProcessingService(mock_storage, mock_llm)
        chat, is_new = await service.process(sample_chat_ingest)

        assert is_new is False
        assert chat == sample_chat_dto
        mock_llm.classify_chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_resolve_title_from_message(
        self,
        mock_storage: AsyncMock,
        mock_llm: AsyncMock,
    ) -> None:
        # Chat without title
        chat_ingest = ChatIngest(
            source="test",
            imported_chat_timestamp=1704067200,
            title=None,
            messages=[
                IngestMessage(
                    role="user",
                    content="What is Python. It's a programming language.",
                    timestamp=1704067200,
                    chat_id="c1",
                ),
            ],
        )

        service = ChatProcessingService(mock_storage, mock_llm)
        chat, _ = await service.process(chat_ingest)

        # Should use first sentence
        assert chat.title == "What is Python"


class TestMessageBuilder:
    """Tests for MessageBuilder service."""

    def test_build_paired_messages(self) -> None:
        messages = [
            IngestMessage(role="user", content="Hello", timestamp=1, chat_id="c1"),
            IngestMessage(role="assistant", content="Hi!", timestamp=2, chat_id="c1"),
            IngestMessage(role="user", content="Bye", timestamp=3, chat_id="c1"),
            IngestMessage(role="assistant", content="Goodbye!", timestamp=4, chat_id="c1"),
        ]

        builder = MessageBuilder()
        result = builder.build(messages, "c1")

        assert len(result) == 2
        assert result[0].user_content == "Hello"
        assert result[0].assistant_content == "Hi!"
        assert result[1].user_content == "Bye"
        assert result[1].assistant_content == "Goodbye!"

    def test_build_standalone_user_message(self) -> None:
        messages = [
            IngestMessage(role="user", content="Hello", timestamp=1, chat_id="c1"),
            IngestMessage(role="user", content="Anyone there?", timestamp=2, chat_id="c1"),
        ]

        builder = MessageBuilder()
        result = builder.build(messages, "c1")

        # First user message becomes standalone, second has no assistant
        assert len(result) == 2
        assert result[0].user_content == "Hello"
        assert result[0].assistant_content == ""

    def test_build_empty_input(self) -> None:
        builder = MessageBuilder()
        result = builder.build([], "c1")
        assert len(result) == 0

    def test_build_sorts_by_timestamp(self) -> None:
        messages = [
            IngestMessage(role="assistant", content="Hi!", timestamp=2, chat_id="c1"),
            IngestMessage(role="user", content="Hello", timestamp=1, chat_id="c1"),
        ]

        builder = MessageBuilder()
        result = builder.build(messages, "c1")

        # Should pair correctly despite input order
        assert result[0].user_content == "Hello"
        assert result[0].assistant_content == "Hi!"


class TestDedupService:
    """Tests for DedupService."""

    @pytest.mark.asyncio
    async def test_new_topic(
        self,
        mock_embedding: AsyncMock,
        mock_storage: AsyncMock,
    ) -> None:
        # No similar topics found
        mock_storage.find_similar_topics.return_value = []

        service = DedupService(mock_embedding, mock_storage)
        result = await service.check_duplicate([0.1] * 1536)

        assert result.action == DedupAction.NEW
        assert result.existing_topic is None

    @pytest.mark.asyncio
    async def test_merge_high_similarity(
        self,
        mock_embedding: AsyncMock,
        mock_storage: AsyncMock,
        sample_topic_dto: TopicDTO,
    ) -> None:
        # High similarity match (>= 0.92)
        mock_storage.find_similar_topics.return_value = [(sample_topic_dto, 0.95)]

        service = DedupService(mock_embedding, mock_storage)
        result = await service.check_duplicate([0.1] * 1536)

        assert result.action == DedupAction.MERGE
        assert result.existing_topic == sample_topic_dto
        assert result.similarity == 0.95

    @pytest.mark.asyncio
    async def test_soft_match_medium_similarity(
        self,
        mock_embedding: AsyncMock,
        mock_storage: AsyncMock,
        sample_topic_dto: TopicDTO,
    ) -> None:
        # Medium similarity (0.80-0.92)
        mock_storage.find_similar_topics.return_value = [(sample_topic_dto, 0.85)]

        service = DedupService(mock_embedding, mock_storage)
        result = await service.check_duplicate([0.1] * 1536)

        assert result.action == DedupAction.SOFT_MATCH
        assert result.existing_topic == sample_topic_dto

    @pytest.mark.asyncio
    async def test_threshold_boundaries(
        self,
        mock_embedding: AsyncMock,
        mock_storage: AsyncMock,
        sample_topic_dto: TopicDTO,
    ) -> None:
        service = DedupService(mock_embedding, mock_storage)

        # Exactly at high threshold
        mock_storage.find_similar_topics.return_value = [(sample_topic_dto, 0.92)]
        result = await service.check_duplicate([0.1] * 1536)
        assert result.action == DedupAction.MERGE

        # Just below high threshold
        mock_storage.find_similar_topics.return_value = [(sample_topic_dto, 0.919)]
        result = await service.check_duplicate([0.1] * 1536)
        assert result.action == DedupAction.SOFT_MATCH

        # Exactly at low threshold
        mock_storage.find_similar_topics.return_value = [(sample_topic_dto, 0.80)]
        result = await service.check_duplicate([0.1] * 1536)
        assert result.action == DedupAction.SOFT_MATCH
