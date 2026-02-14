"""BotKnows orchestrator for high-level knowledge base operations.

This module provides the main entry point for the bot_knows package,
orchestrating all services for chat ingestion and knowledge retrieval.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bot_knows.config import BotKnowsConfig
from bot_knows.importers.base import ChatImportAdapter
from bot_knows.interfaces.embedding import EmbeddingServiceInterface
from bot_knows.interfaces.graph import GraphServiceInterface
from bot_knows.interfaces.llm import LLMInterface
from bot_knows.interfaces.storage import StorageInterface
from bot_knows.logging import get_logger
from bot_knows.models.chat import ChatDTO
from bot_knows.models.ingest import ChatIngest
from bot_knows.models.message import MessageDTO
from bot_knows.models.recall import TopicRecallStateDTO
from bot_knows.models.topic import TopicEvidenceDTO
from bot_knows.services.chat_processing import ChatProcessingService
from bot_knows.services.dedup_service import DedupAction, DedupService
from bot_knows.services.graph_service import GraphService
from bot_knows.services.message_builder import MessageBuilder
from bot_knows.services.recall_service import RecallService
from bot_knows.services.topic_extraction import TopicExtractionService

__all__ = ["BotKnows", "InsertResult"]

logger = get_logger(__name__)


@dataclass
class InsertResult:
    """Statistics from chat insertion."""

    chats_processed: int = 0
    chats_new: int = 0
    chats_updated: int = 0
    chats_skipped: int = 0
    messages_created: int = 0
    topics_created: int = 0
    topics_merged: int = 0
    topics_soft_matched: int = 0
    errors: list[str] = field(default_factory=list)


class BotKnows:
    """Main orchestrator and retriver for bot_knows knowledge base.

    Accepts implementation classes. Config is loaded from .env automatically.
    For custom implementations, set config_class = None and pass custom_config dict.

    Example:
        async with BotKnows(
            storage_class=MongoStorageRepository,
            graphdb_class=Neo4jGraphRepository,
            llm_class=OpenAIProvider,
        ) as bk:
            result = await bk.insert_chats("export.json", ChatGPTAdapter)
    """

    def __init__(
        self,
        storage_class: type[StorageInterface],
        graphdb_class: type[GraphServiceInterface],
        llm_class: type[LLMInterface],
        embedding_class: type[EmbeddingServiceInterface] | None = None,
        *,
        storage_custom_config: dict[str, Any] | None = None,
        graphdb_custom_config: dict[str, Any] | None = None,
        llm_custom_config: dict[str, Any] | None = None,
        embedding_custom_config: dict[str, Any] | None = None,
    ) -> None:
        """Initialize BotKnows with implementation classes.

        Args:
            storage_class: Storage implementation class
            graphdb_class: Graph DB implementation class
            llm_class: LLM implementation class
            embedding_class: Embedding implementation class (defaults to llm_class)
            storage_custom_config: Custom config dict if storage_class.config_class is None
            graphdb_custom_config: Custom config dict if graphdb_class.config_class is None
            llm_custom_config: Custom config dict if llm_class.config_class is None
            embedding_custom_config: Custom config dict if embedding_class.config_class is None
        """
        self._config = BotKnowsConfig()  # Loads from .env

        self._storage_class = storage_class
        self._graphdb_class = graphdb_class
        self._llm_class = llm_class
        self._embedding_class = embedding_class or llm_class

        self._storage_custom_config = storage_custom_config
        self._graphdb_custom_config = graphdb_custom_config
        self._llm_custom_config = llm_custom_config
        self._embedding_custom_config = embedding_custom_config

        # Instances (created on connect)
        self._storage: StorageInterface | None = None
        self._graph: GraphServiceInterface | None = None
        self._llm: LLMInterface | None = None
        self._embedding: EmbeddingServiceInterface | None = None

        # Services (wired on connect)
        self._chat_processor: ChatProcessingService | None = None
        self._message_builder = MessageBuilder()
        self._topic_extractor: TopicExtractionService | None = None
        self._dedup_service: DedupService | None = None
        self._graph_service: GraphService | None = None
        self._recall_service: RecallService | None = None

        self._connected = False

    async def _instantiate_class(
        self,
        cls: type,
        custom_config: dict[str, Any] | None,
    ) -> Any:
        """Instantiate an implementation class.

        If cls.config_class is set, instantiate config (loads from .env).
        If cls.config_class is None, use custom_config dict.
        """
        config_class = getattr(cls, "config_class", None)

        if config_class is None:
            # Custom implementation - use dict
            if custom_config is None:
                raise ValueError(
                    f"{cls.__name__} has config_class=None but no custom_config provided"
                )
            return await cls.from_dict(custom_config)
        else:
            # Standard implementation - instantiate settings (loads from .env)
            config = config_class()
            return await cls.from_config(config)

    async def _connect(self) -> None:
        """Initialize connections and services."""
        if self._connected:
            return

        # Instantiate implementations
        self._storage = await self._instantiate_class(
            self._storage_class, self._storage_custom_config
        )
        self._graph = await self._instantiate_class(
            self._graphdb_class, self._graphdb_custom_config
        )
        self._llm = await self._instantiate_class(self._llm_class, self._llm_custom_config)

        if self._embedding_class is self._llm_class:
            self._embedding = self._llm  # type: ignore[assignment]
        else:
            self._embedding = await self._instantiate_class(
                self._embedding_class, self._embedding_custom_config
            )

        # Wire services
        self._chat_processor = ChatProcessingService(self._storage, self._llm)
        self._topic_extractor = TopicExtractionService(self._llm, self._embedding)
        self._dedup_service = DedupService(
            self._embedding,
            self._storage,
            high_threshold=self._config.dedup_high_threshold,
            low_threshold=self._config.dedup_low_threshold,
        )
        self._graph_service = GraphService(self._graph)
        self._recall_service = RecallService(
            self._storage,
            self._graph,
            stability_k=self._config.recall_stability_k,
            semantic_boost=self._config.recall_semantic_boost,
        )

        self._connected = True
        logger.info("bot_knows_connected")

    async def _disconnect(self) -> None:
        """Close all connections."""
        if self._storage and hasattr(self._storage, "close"):
            await self._storage.close()
        if self._graph and hasattr(self._graph, "close"):
            await self._graph.close()
        if self._llm and hasattr(self._llm, "close"):
            await self._llm.close()
        if (
            self._embedding
            and self._embedding is not self._llm
            and hasattr(self._embedding, "close")
        ):
            await self._embedding.close()

        self._connected = False
        logger.info("bot_knows_disconnected")

    async def __aenter__(self) -> "BotKnows":
        """Async context manager entry - connects automatically."""
        await self._connect()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        """Async context manager exit - disconnects automatically."""
        await self._disconnect()

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("BotKnows not connected. Use 'async with BotKnows(...) as bk:'")

    # === MAIN WORKFLOW ===

    async def insert_chats(
        self,
        chats: dict[str, Any] | str | Path,
        adapter_class: type[ChatImportAdapter],
    ) -> InsertResult:
        """Ingest chats through the complete processing pipeline.

        Args:
            chats: Raw export data (dict), JSON string, or path to JSON file
            adapter_class: Import adapter class to use for parsing

        Returns:
            InsertResult with statistics
        """
        self._ensure_connected()

        # Parse input
        adapter = adapter_class()
        chat_ingests: list[ChatIngest]

        if isinstance(chats, Path):
            chat_ingests = adapter.parse_file(chats)
        elif isinstance(chats, str):
            path = Path(chats)
            if path.exists():
                chat_ingests = adapter.parse_file(path)
            else:
                chat_ingests = adapter.parse_string(chats)
        else:
            chat_ingests = adapter.parse(chats)

        result = InsertResult()

        for chat_ingest in chat_ingests:
            try:
                await self._process_single_chat(chat_ingest, result)
            except Exception as e:
                logger.error("chat_processing_failed", error=str(e))
                result.errors.append(f"Chat '{chat_ingest.title}': {e}")

        logger.info(
            "insert_chats_completed",
            chats_processed=result.chats_processed,
            chats_new=result.chats_new,
            chats_updated=result.chats_updated,
            topics_created=result.topics_created,
        )

        return result

    async def _process_single_chat(
        self,
        chat_ingest: ChatIngest,
        result: InsertResult,
    ) -> None:
        """Process a single chat through the pipeline."""
        assert self._chat_processor is not None
        assert self._storage is not None
        assert self._graph_service is not None

        result.chats_processed += 1

        # Step 1: Process chat (identity, classification, persistence)
        chat, is_new = await self._chat_processor.process(chat_ingest)

        if is_new:
            # New chat - full processing
            result.chats_new += 1
            messages = self._message_builder.build(chat_ingest.messages, chat)

            for message in messages:
                await self._storage.save_message(message)
                result.messages_created += 1

            await self._graph_service.add_chat_with_messages(chat.id, messages)

            for message in messages:
                await self._process_message_topics(message, result)
        else:
            # Existing chat - check for incremental update
            await self._process_incremental_chat(chat, chat_ingest, result)

    async def _process_incremental_chat(
        self,
        chat: ChatDTO,
        chat_ingest: ChatIngest,
        result: InsertResult,
    ) -> None:
        """Process incremental updates for an existing chat."""
        assert self._storage is not None
        assert self._graph_service is not None

        # Get existing messages from storage
        existing_messages = await self._storage.get_messages_for_chat(chat.id)
        existing_message_ids = {msg.message_id for msg in existing_messages}

        # Build all messages from import
        all_messages = self._message_builder.build(chat_ingest.messages, chat)

        # If DB has same or more messages, skip
        if len(existing_messages) >= len(all_messages):
            result.chats_skipped += 1
            return

        # Find new messages (not in existing)
        new_messages = [msg for msg in all_messages if msg.message_id not in existing_message_ids]

        if not new_messages:
            result.chats_skipped += 1
            return

        result.chats_updated += 1

        # Find the last existing message ID for graph chaining
        last_existing_message_id = (
            existing_messages[-1].message_id if existing_messages else None
        )

        # Save new messages to storage
        for message in new_messages:
            await self._storage.save_message(message)
            result.messages_created += 1

        # Add new messages to graph
        await self._graph_service.add_messages_to_chat(
            chat.id, new_messages, last_existing_message_id
        )

        # Process topics for new messages
        for message in new_messages:
            await self._process_message_topics(message, result)

        logger.info(
            "chat_incrementally_updated",
            chat_id=chat.id,
            new_messages=len(new_messages),
        )

    async def _process_message_topics(
        self,
        message: MessageDTO,
        result: InsertResult,
    ) -> None:
        """Extract and process topics from a message."""
        assert self._topic_extractor is not None
        assert self._dedup_service is not None
        assert self._storage is not None
        assert self._graph_service is not None
        assert self._recall_service is not None

        candidates = await self._topic_extractor.extract(message)

        for candidate in candidates:
            dedup_result = await self._dedup_service.check_duplicate(candidate.embedding)

            if dedup_result.action == DedupAction.MERGE:
                assert dedup_result.existing_topic is not None
                topic, evidence = await self._topic_extractor.create_evidence_for_existing(
                    candidate, dedup_result.existing_topic
                )
                await self._storage.update_topic(topic)
                await self._storage.append_evidence(evidence)
                await self._graph_service.add_evidence_to_existing_topic(topic, evidence)
                result.topics_merged += 1

            elif dedup_result.action == DedupAction.SOFT_MATCH:
                assert dedup_result.existing_topic is not None
                topic, evidence = await self._topic_extractor.create_topic_and_evidence(candidate)
                await self._storage.save_topic(topic)
                await self._storage.append_evidence(evidence)
                await self._graph_service.add_topic_with_evidence(topic, evidence)
                await self._graph_service.create_topic_relation(
                    topic.topic_id,
                    dedup_result.existing_topic.topic_id,
                    dedup_result.similarity,
                )
                result.topics_soft_matched += 1
                result.topics_created += 1

            else:  # NEW
                topic, evidence = await self._topic_extractor.create_topic_and_evidence(candidate)
                await self._storage.save_topic(topic)
                await self._storage.append_evidence(evidence)
                await self._graph_service.add_topic_with_evidence(topic, evidence)
                result.topics_created += 1

            # Reinforce recall state
            await self._recall_service.reinforce(
                topic.topic_id,
                confidence=candidate.confidence,
                context="passive",
            )

    # === RETRIEVAL METHODS ===

    async def get_messages_for_chat(self, chat_id: str) -> list[MessageDTO]:
        """Get all messages for a chat."""
        self._ensure_connected()
        assert self._storage is not None
        return await self._storage.get_messages_for_chat(chat_id)

    async def get_related_topics(self, topic_id: str, limit: int = 10) -> list[tuple[str, float]]:
        """Get topics related to a given topic."""
        self._ensure_connected()
        assert self._graph is not None
        return await self._graph.get_related_topics(topic_id, limit)

    async def get_topic_evidence(self, topic_id: str) -> list[TopicEvidenceDTO]:
        """Get all evidence for a topic."""
        self._ensure_connected()
        assert self._storage is not None
        return await self._storage.get_evidence_for_topic(topic_id)

    async def get_chat_topics(self, chat_id: str) -> list[str]:
        """Get all topic IDs associated with a chat's messages."""
        self._ensure_connected()
        assert self._graph is not None
        return await self._graph.get_chat_topics(chat_id)

    async def get_recall_state(self, topic_id: str) -> TopicRecallStateDTO | None:
        """Get recall state for a topic."""
        self._ensure_connected()
        assert self._storage is not None
        return await self._storage.get_recall_state(topic_id)

    async def get_due_topics(self, threshold: float = 0.3) -> list[TopicRecallStateDTO]:
        """Get topics due for recall review."""
        self._ensure_connected()
        assert self._storage is not None
        return await self._storage.get_due_topics(threshold)

    async def get_all_recall_states(self) -> list[TopicRecallStateDTO]:
        """Get all recall states."""
        self._ensure_connected()
        assert self._storage is not None
        return await self._storage.get_all_recall_states()
