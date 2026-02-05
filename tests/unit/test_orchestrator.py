"""Unit tests for BotKnows orchestrator."""

from typing import Any, Self
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot_knows.interfaces.embedding import EmbeddingServiceInterface
from bot_knows.interfaces.graph import GraphServiceInterface
from bot_knows.interfaces.llm import LLMInterface
from bot_knows.interfaces.storage import StorageInterface
from bot_knows.models.chat import ChatCategory, ChatDTO
from bot_knows.models.ingest import ChatIngest, IngestMessage
from bot_knows.orchestrator import BotKnows, InsertResult


# Mock implementation classes for testing
class MockStorageImpl(StorageInterface):
    """Mock storage implementation with config_class set."""

    config_class = MagicMock()  # Simulates having a config class

    def __init__(self) -> None:
        self._mock = AsyncMock()
        self._mock.get_chat.return_value = None
        self._mock.chat_exists.return_value = False
        self._mock.save_chat.return_value = "test-chat-id"
        self._mock.save_message.return_value = "test-message-id"
        self._mock.save_topic.return_value = "test-topic-id"
        self._mock.get_topic.return_value = None
        self._mock.find_similar_topics.return_value = []
        self._mock.get_recall_state.return_value = None
        self._mock.get_due_topics.return_value = []
        self._mock.get_messages_for_chat.return_value = []
        self._mock.get_evidence_for_topic.return_value = []
        self._mock.get_all_recall_states.return_value = []

    @classmethod
    async def from_config(cls, config: Any) -> Self:
        return cls()

    @classmethod
    async def from_dict(cls, config: dict[str, Any]) -> Self:
        return cls()

    async def close(self) -> None:
        pass

    # Delegate all interface methods to mock
    async def save_chat(self, chat: ChatDTO) -> str:
        return await self._mock.save_chat(chat)

    async def get_chat(self, chat_id: str) -> ChatDTO | None:
        return await self._mock.get_chat(chat_id)

    async def chat_exists(self, chat_id: str) -> bool:
        return await self._mock.chat_exists(chat_id)

    async def find_chats_by_source(self, source: str) -> list[ChatDTO]:
        return await self._mock.find_chats_by_source(source)

    async def save_message(self, message: Any) -> str:
        return await self._mock.save_message(message)

    async def get_message(self, message_id: str) -> Any:
        return await self._mock.get_message(message_id)

    async def get_messages_for_chat(self, chat_id: str) -> list[Any]:
        return await self._mock.get_messages_for_chat(chat_id)

    async def save_topic(self, topic: Any) -> str:
        return await self._mock.save_topic(topic)

    async def get_topic(self, topic_id: str) -> Any:
        return await self._mock.get_topic(topic_id)

    async def update_topic(self, topic: Any) -> None:
        await self._mock.update_topic(topic)

    async def find_similar_topics(
        self, embedding: list[float], threshold: float
    ) -> list[tuple[Any, float]]:
        return await self._mock.find_similar_topics(embedding, threshold)

    async def get_all_topics(self, limit: int = 1000) -> list[Any]:
        return await self._mock.get_all_topics(limit)

    async def append_evidence(self, evidence: Any) -> str:
        return await self._mock.append_evidence(evidence)

    async def get_evidence_for_topic(self, topic_id: str) -> list[Any]:
        return await self._mock.get_evidence_for_topic(topic_id)

    async def save_recall_state(self, state: Any) -> None:
        await self._mock.save_recall_state(state)

    async def get_recall_state(self, topic_id: str) -> Any:
        return await self._mock.get_recall_state(topic_id)

    async def get_due_topics(self, threshold: float) -> list[Any]:
        return await self._mock.get_due_topics(threshold)

    async def get_all_recall_states(self) -> list[Any]:
        return await self._mock.get_all_recall_states()


class MockGraphImpl(GraphServiceInterface):
    """Mock graph implementation with config_class set."""

    config_class = MagicMock()

    def __init__(self) -> None:
        self._mock = AsyncMock()
        self._mock.get_related_topics.return_value = []
        self._mock.get_topic_evidence.return_value = []
        self._mock.get_chat_topics.return_value = []

    @classmethod
    async def from_config(cls, config: Any) -> Self:
        return cls()

    @classmethod
    async def from_dict(cls, config: dict[str, Any]) -> Self:
        return cls()

    async def close(self) -> None:
        pass

    async def create_chat_node(self, chat: Any) -> str:
        return await self._mock.create_chat_node(chat)

    async def create_message_node(self, message: Any) -> str:
        return await self._mock.create_message_node(message)

    async def create_topic_node(self, topic: Any) -> str:
        return await self._mock.create_topic_node(topic)

    async def update_topic_node(self, topic: Any) -> None:
        await self._mock.update_topic_node(topic)

    async def create_is_part_of_edge(self, message_id: str, chat_id: str) -> None:
        await self._mock.create_is_part_of_edge(message_id, chat_id)

    async def create_follows_after_edge(self, message_id: str, previous_message_id: str) -> None:
        await self._mock.create_follows_after_edge(message_id, previous_message_id)

    async def create_is_supported_by_edge(
        self, topic_id: str, message_id: str, evidence: Any
    ) -> None:
        await self._mock.create_is_supported_by_edge(topic_id, message_id, evidence)

    async def create_relates_to_edge(
        self, topic_id: str, related_topic_id: str, similarity: float
    ) -> None:
        await self._mock.create_relates_to_edge(topic_id, related_topic_id, similarity)

    async def get_messages_for_chat(self, chat_id: str) -> list[Any]:
        return await self._mock.get_messages_for_chat(chat_id)

    async def get_related_topics(self, topic_id: str, limit: int = 10) -> list[tuple[str, float]]:
        return await self._mock.get_related_topics(topic_id, limit)

    async def get_topic_evidence(self, topic_id: str) -> list[dict[str, Any]]:
        return await self._mock.get_topic_evidence(topic_id)

    async def get_chat_topics(self, chat_id: str) -> list[str]:
        return await self._mock.get_chat_topics(chat_id)


class MockLLMImpl(LLMInterface, EmbeddingServiceInterface):
    """Mock LLM implementation with config_class set."""

    config_class = MagicMock()

    def __init__(self) -> None:
        self._mock = AsyncMock()
        self._mock.classify_chat.return_value = (ChatCategory.CODING, ["python"])
        self._mock.extract_topics.return_value = [("Python", 0.9)]
        self._mock.normalize_topic_name.return_value = "Python"
        self._mock.embed.return_value = [0.1] * 1536
        self._mock.embed_batch.return_value = [[0.1] * 1536]
        self._mock.similarity.return_value = 0.95

    @classmethod
    async def from_config(cls, config: Any) -> Self:
        return cls()

    @classmethod
    async def from_dict(cls, config: dict[str, Any]) -> Self:
        return cls()

    async def close(self) -> None:
        pass

    async def classify_chat(
        self, first_pair: tuple[str, str], last_pair: tuple[str, str]
    ) -> tuple[ChatCategory, list[str]]:
        return await self._mock.classify_chat(first_pair, last_pair)

    async def extract_topics(
        self, user_content: str, assistant_content: str
    ) -> list[tuple[str, float]]:
        return await self._mock.extract_topics(user_content, assistant_content)

    async def normalize_topic_name(self, extracted_name: str) -> str:
        return await self._mock.normalize_topic_name(extracted_name)

    async def embed(self, text: str) -> list[float]:
        return await self._mock.embed(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return await self._mock.embed_batch(texts)

    async def similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        return await self._mock.similarity(embedding1, embedding2)


# Custom implementation with config_class = None
class CustomStorageImpl(StorageInterface):
    """Custom storage implementation with config_class = None."""

    config_class = None  # Signals BotKnows to use custom_config dict

    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self._mock = AsyncMock()
        self._mock.get_chat.return_value = None
        self._mock.find_similar_topics.return_value = []
        self._mock.get_recall_state.return_value = None
        self._mock.get_due_topics.return_value = []
        self._mock.get_messages_for_chat.return_value = []
        self._mock.get_evidence_for_topic.return_value = []
        self._mock.get_all_recall_states.return_value = []

    @classmethod
    async def from_dict(cls, config: dict[str, Any]) -> Self:
        return cls(host=config["host"], port=config["port"])

    async def close(self) -> None:
        pass

    # Implement required interface methods (delegating to mock)
    async def save_chat(self, chat: Any) -> str:
        return await self._mock.save_chat(chat)

    async def get_chat(self, chat_id: str) -> Any:
        return await self._mock.get_chat(chat_id)

    async def chat_exists(self, chat_id: str) -> bool:
        return await self._mock.chat_exists(chat_id)

    async def find_chats_by_source(self, source: str) -> list[Any]:
        return await self._mock.find_chats_by_source(source)

    async def save_message(self, message: Any) -> str:
        return await self._mock.save_message(message)

    async def get_message(self, message_id: str) -> Any:
        return await self._mock.get_message(message_id)

    async def get_messages_for_chat(self, chat_id: str) -> list[Any]:
        return await self._mock.get_messages_for_chat(chat_id)

    async def save_topic(self, topic: Any) -> str:
        return await self._mock.save_topic(topic)

    async def get_topic(self, topic_id: str) -> Any:
        return await self._mock.get_topic(topic_id)

    async def update_topic(self, topic: Any) -> None:
        await self._mock.update_topic(topic)

    async def find_similar_topics(
        self, embedding: list[float], threshold: float
    ) -> list[tuple[Any, float]]:
        return await self._mock.find_similar_topics(embedding, threshold)

    async def get_all_topics(self, limit: int = 1000) -> list[Any]:
        return await self._mock.get_all_topics(limit)

    async def append_evidence(self, evidence: Any) -> str:
        return await self._mock.append_evidence(evidence)

    async def get_evidence_for_topic(self, topic_id: str) -> list[Any]:
        return await self._mock.get_evidence_for_topic(topic_id)

    async def save_recall_state(self, state: Any) -> None:
        await self._mock.save_recall_state(state)

    async def get_recall_state(self, topic_id: str) -> Any:
        return await self._mock.get_recall_state(topic_id)

    async def get_due_topics(self, threshold: float) -> list[Any]:
        return await self._mock.get_due_topics(threshold)

    async def get_all_recall_states(self) -> list[Any]:
        return await self._mock.get_all_recall_states()


class TestBotKnowsInit:
    """Tests for BotKnows initialization."""

    def test_init_stores_classes(self) -> None:
        """Test that BotKnows stores implementation classes."""
        bk = BotKnows(
            storage_class=MockStorageImpl,
            graphdb_class=MockGraphImpl,
            llm_class=MockLLMImpl,
        )

        assert bk._storage_class is MockStorageImpl
        assert bk._graphdb_class is MockGraphImpl
        assert bk._llm_class is MockLLMImpl
        assert bk._connected is False

    def test_init_defaults_embedding_to_llm(self) -> None:
        """Test that embedding_class defaults to llm_class."""
        bk = BotKnows(
            storage_class=MockStorageImpl,
            graphdb_class=MockGraphImpl,
            llm_class=MockLLMImpl,
        )

        assert bk._embedding_class is MockLLMImpl


class TestBotKnowsContextManager:
    """Tests for BotKnows context manager behavior."""

    @pytest.mark.asyncio
    async def test_context_manager_connects_and_disconnects(self) -> None:
        """Test that context manager connects on enter and disconnects on exit."""
        async with BotKnows(
            storage_class=MockStorageImpl,
            graphdb_class=MockGraphImpl,
            llm_class=MockLLMImpl,
        ) as bk:
            assert bk._connected is True
            assert bk._storage is not None
            assert bk._graph is not None
            assert bk._llm is not None

        # After exiting context, should be disconnected
        assert bk._connected is False

    @pytest.mark.asyncio
    async def test_services_are_wired_after_connect(self) -> None:
        """Test that services are properly wired after connect."""
        async with BotKnows(
            storage_class=MockStorageImpl,
            graphdb_class=MockGraphImpl,
            llm_class=MockLLMImpl,
        ) as bk:
            assert bk._chat_processor is not None
            assert bk._topic_extractor is not None
            assert bk._dedup_service is not None
            assert bk._graph_service is not None
            assert bk._recall_service is not None


class TestBotKnowsCustomConfig:
    """Tests for custom implementations with config_class = None."""

    @pytest.mark.asyncio
    async def test_custom_storage_with_dict_config(self) -> None:
        """Test that custom storage with config_class=None receives dict config."""
        custom_config = {"host": "localhost", "port": 5432}

        async with BotKnows(
            storage_class=CustomStorageImpl,
            storage_custom_config=custom_config,
            graphdb_class=MockGraphImpl,
            llm_class=MockLLMImpl,
        ) as bk:
            assert isinstance(bk._storage, CustomStorageImpl)
            assert bk._storage.host == "localhost"
            assert bk._storage.port == 5432

    @pytest.mark.asyncio
    async def test_custom_storage_without_config_raises_error(self) -> None:
        """Test that custom storage without custom_config raises ValueError."""
        with pytest.raises(ValueError, match="has config_class=None but no custom_config"):
            async with BotKnows(
                storage_class=CustomStorageImpl,
                # Missing storage_custom_config
                graphdb_class=MockGraphImpl,
                llm_class=MockLLMImpl,
            ):
                pass


class TestBotKnowsInsertChats:
    """Tests for insert_chats workflow."""

    @pytest.fixture
    def sample_chat_ingest(self) -> ChatIngest:
        """Create sample ChatIngest for testing."""
        return ChatIngest(
            source="test",
            imported_chat_timestamp=1704067200,
            title="Test Chat",
            messages=[
                IngestMessage(
                    role="user",
                    content="Hello",
                    timestamp=1704067200,
                    chat_id="test-1",
                ),
                IngestMessage(
                    role="assistant",
                    content="Hi there!",
                    timestamp=1704067201,
                    chat_id="test-1",
                ),
            ],
            provider="test",
            conversation_id="test-1",
        )

    @pytest.mark.asyncio
    async def test_insert_chats_returns_result(self, sample_chat_ingest: ChatIngest) -> None:
        """Test that insert_chats returns InsertResult."""
        # Create mock adapter class
        mock_adapter_class = MagicMock()
        mock_adapter_instance = MagicMock()
        mock_adapter_instance.parse.return_value = [sample_chat_ingest]
        mock_adapter_class.return_value = mock_adapter_instance

        async with BotKnows(
            storage_class=MockStorageImpl,
            graphdb_class=MockGraphImpl,
            llm_class=MockLLMImpl,
        ) as bk:
            result = await bk.insert_chats({"test": "data"}, mock_adapter_class)

            assert isinstance(result, InsertResult)
            assert result.chats_processed == 1
            assert result.chats_new == 1
            assert result.messages_created >= 1

    @pytest.mark.asyncio
    async def test_insert_chats_skips_existing_chat(self, sample_chat_ingest: ChatIngest) -> None:
        """Test that existing chats are skipped."""
        mock_adapter_class = MagicMock()
        mock_adapter_instance = MagicMock()
        mock_adapter_instance.parse.return_value = [sample_chat_ingest]
        mock_adapter_class.return_value = mock_adapter_instance

        async with BotKnows(
            storage_class=MockStorageImpl,
            graphdb_class=MockGraphImpl,
            llm_class=MockLLMImpl,
        ) as bk:
            # Simulate existing chat by patching the storage mock
            assert isinstance(bk._storage, MockStorageImpl)
            bk._storage._mock.get_chat.return_value = ChatDTO(
                id="existing",
                title="Existing",
                source="test",
                category=ChatCategory.GENERAL,
                tags=[],
                created_on=1704067200,
            )

            result = await bk.insert_chats({"test": "data"}, mock_adapter_class)

            assert result.chats_processed == 1
            assert result.chats_skipped == 1
            assert result.chats_new == 0

    @pytest.mark.asyncio
    async def test_insert_chats_not_connected_raises_error(self) -> None:
        """Test that insert_chats raises error when not connected."""
        bk = BotKnows(
            storage_class=MockStorageImpl,
            graphdb_class=MockGraphImpl,
            llm_class=MockLLMImpl,
        )

        mock_adapter_class = MagicMock()

        with pytest.raises(RuntimeError, match="not connected"):
            await bk.insert_chats({"test": "data"}, mock_adapter_class)


class TestBotKnowsRetrievalMethods:
    """Tests for retrieval methods."""

    @pytest.mark.asyncio
    async def test_get_messages_for_chat(self) -> None:
        """Test get_messages_for_chat delegates to storage."""
        async with BotKnows(
            storage_class=MockStorageImpl,
            graphdb_class=MockGraphImpl,
            llm_class=MockLLMImpl,
        ) as bk:
            result = await bk.get_messages_for_chat("test-chat-id")

            assert isinstance(result, list)
            # Verify the mock was called
            assert isinstance(bk._storage, MockStorageImpl)
            bk._storage._mock.get_messages_for_chat.assert_called_once_with("test-chat-id")

    @pytest.mark.asyncio
    async def test_get_related_topics(self) -> None:
        """Test get_related_topics delegates to graph."""
        async with BotKnows(
            storage_class=MockStorageImpl,
            graphdb_class=MockGraphImpl,
            llm_class=MockLLMImpl,
        ) as bk:
            result = await bk.get_related_topics("test-topic-id", limit=5)

            assert isinstance(result, list)
            assert isinstance(bk._graph, MockGraphImpl)
            bk._graph._mock.get_related_topics.assert_called_once_with("test-topic-id", 5)

    @pytest.mark.asyncio
    async def test_get_chat_topics(self) -> None:
        """Test get_chat_topics delegates to graph."""
        async with BotKnows(
            storage_class=MockStorageImpl,
            graphdb_class=MockGraphImpl,
            llm_class=MockLLMImpl,
        ) as bk:
            result = await bk.get_chat_topics("test-chat-id")

            assert isinstance(result, list)
            assert isinstance(bk._graph, MockGraphImpl)
            bk._graph._mock.get_chat_topics.assert_called_once_with("test-chat-id")

    @pytest.mark.asyncio
    async def test_get_recall_state(self) -> None:
        """Test get_recall_state delegates to storage."""
        async with BotKnows(
            storage_class=MockStorageImpl,
            graphdb_class=MockGraphImpl,
            llm_class=MockLLMImpl,
        ) as bk:
            _ = await bk.get_recall_state("test-topic-id")

            assert isinstance(bk._storage, MockStorageImpl)
            bk._storage._mock.get_recall_state.assert_called_once_with("test-topic-id")

    @pytest.mark.asyncio
    async def test_get_due_topics(self) -> None:
        """Test get_due_topics delegates to storage."""
        async with BotKnows(
            storage_class=MockStorageImpl,
            graphdb_class=MockGraphImpl,
            llm_class=MockLLMImpl,
        ) as bk:
            result = await bk.get_due_topics(threshold=0.5)

            assert isinstance(result, list)
            assert isinstance(bk._storage, MockStorageImpl)
            bk._storage._mock.get_due_topics.assert_called_once_with(0.5)

    @pytest.mark.asyncio
    async def test_get_all_recall_states(self) -> None:
        """Test get_all_recall_states delegates to storage."""
        async with BotKnows(
            storage_class=MockStorageImpl,
            graphdb_class=MockGraphImpl,
            llm_class=MockLLMImpl,
        ) as bk:
            result = await bk.get_all_recall_states()

            assert isinstance(result, list)
            assert isinstance(bk._storage, MockStorageImpl)
            bk._storage._mock.get_all_recall_states.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieval_not_connected_raises_error(self) -> None:
        """Test that retrieval methods raise error when not connected."""
        bk = BotKnows(
            storage_class=MockStorageImpl,
            graphdb_class=MockGraphImpl,
            llm_class=MockLLMImpl,
        )

        with pytest.raises(RuntimeError, match="not connected"):
            await bk.get_messages_for_chat("test-id")


class TestInsertResult:
    """Tests for InsertResult dataclass."""

    def test_default_values(self) -> None:
        """Test InsertResult has correct default values."""
        result = InsertResult()

        assert result.chats_processed == 0
        assert result.chats_new == 0
        assert result.chats_skipped == 0
        assert result.messages_created == 0
        assert result.topics_created == 0
        assert result.topics_merged == 0
        assert result.topics_soft_matched == 0
        assert result.errors == []

    def test_errors_list_is_mutable(self) -> None:
        """Test that errors list can be modified."""
        result = InsertResult()
        result.errors.append("Test error")

        assert len(result.errors) == 1
        assert result.errors[0] == "Test error"
