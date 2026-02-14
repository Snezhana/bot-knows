"""MongoDB repositories for bot_knows.

This module provides repository implementations for MongoDB storage.
"""

from typing import Any, Self

import numpy as np

from bot_knows.config import MongoSettings
from bot_knows.infra.mongo.client import MongoClient
from bot_knows.interfaces.storage import StorageInterface
from bot_knows.logging import get_logger
from bot_knows.models.chat import ChatCategory, ChatDTO
from bot_knows.models.message import MessageDTO
from bot_knows.models.recall import TopicRecallStateDTO
from bot_knows.models.topic import TopicDTO, TopicEvidenceDTO

__all__ = [
    "MongoStorageRepository",
]

logger = get_logger(__name__)


class MongoStorageRepository(StorageInterface):
    """MongoDB implementation of StorageInterface.

    Provides CRUD operations for chats, messages, topics,
    evidence, and recall states.
    """

    config_class = MongoSettings

    def __init__(
        self,
        client: MongoClient,
        vector_search_enabled: bool = False,
        vector_search_index_name: str = "topic_embedding_index",
        vector_search_num_candidates: int = 100,
    ) -> None:
        """Initialize repository with MongoDB client.

        Args:
            client: Connected MongoClient instance
            vector_search_enabled: Whether Atlas Vector Search is available
            vector_search_index_name: Name of the vector search index
            vector_search_num_candidates: Number of candidates for ANN search
        """
        self._client = client
        self._owns_client = False
        self._vector_search_enabled = vector_search_enabled
        self._vector_search_index_name = vector_search_index_name
        self._vector_search_num_candidates = vector_search_num_candidates

    @classmethod
    async def from_config(cls, config: MongoSettings) -> Self:
        """Factory method for BotKnows instantiation.

        Creates a MongoClient, connects, creates indexes, and returns repository.

        Args:
            config: MongoDB settings

        Returns:
            Connected MongoStorageRepository instance
        """
        from bot_knows.config import LLMSettings

        client = MongoClient(config)
        await client.connect()
        await client.create_indexes()

        # Try to create vector search index if enabled and on Atlas
        vector_search_enabled = False
        if config.vector_search_enabled and client.vector_search_available:
            llm_settings = LLMSettings()
            vector_search_enabled = await client.create_vector_search_index(
                index_name=config.vector_search_index_name,
                dimensions=llm_settings.embedding_dimensions,
            )

        instance = cls(
            client,
            vector_search_enabled=vector_search_enabled,
            vector_search_index_name=config.vector_search_index_name,
            vector_search_num_candidates=config.vector_search_num_candidates,
        )
        instance._owns_client = True
        return instance

    @classmethod
    async def from_dict(cls, config: dict[str, Any]) -> Self:
        """Factory method for custom config dict.

        Args:
            config: Dictionary with MongoDB settings

        Returns:
            Connected MongoStorageRepository instance
        """
        settings = MongoSettings(**config)
        return await cls.from_config(settings)

    async def close(self) -> None:
        """Close owned resources."""
        if self._owns_client and self._client:
            await self._client.disconnect()

    # Chat operations
    async def save_chat(self, chat: ChatDTO) -> str:
        """Save or update a chat."""
        doc = self._chat_to_doc(chat)
        await self._client.chats.replace_one(
            {"id": chat.id},
            doc,
            upsert=True,
        )
        return chat.id

    async def get_chat(self, chat_id: str) -> ChatDTO | None:
        """Get a chat by ID."""
        doc = await self._client.chats.find_one({"id": chat_id})
        return self._doc_to_chat(doc) if doc else None

    async def chat_exists(self, chat_id: str) -> bool:
        """Check if a chat exists."""
        count = await self._client.chats.count_documents({"id": chat_id}, limit=1)
        return count > 0

    async def find_chats_by_source(self, source: str) -> list[ChatDTO]:
        """Find all chats from a source."""
        cursor = self._client.chats.find({"source": source})
        return [self._doc_to_chat(doc) async for doc in cursor]

    # Message operations
    async def save_message(self, message: MessageDTO) -> str:
        """Save or update a message."""
        doc = self._message_to_doc(message)
        await self._client.messages.replace_one(
            {"message_id": message.message_id},
            doc,
            upsert=True,
        )
        return message.message_id

    async def get_message(self, message_id: str) -> MessageDTO | None:
        """Get a message by ID."""
        doc = await self._client.messages.find_one({"message_id": message_id})
        return self._doc_to_message(doc) if doc else None

    async def get_messages_for_chat(self, chat_id: str) -> list[MessageDTO]:
        """Get all messages for a chat, ordered by timestamp."""
        cursor = self._client.messages.find({"chat_id": chat_id}).sort("created_on", 1)
        return [self._doc_to_message(doc) async for doc in cursor]

    # Topic operations
    async def save_topic(self, topic: TopicDTO) -> str:
        """Save or update a topic."""
        doc = self._topic_to_doc(topic)
        await self._client.topics.replace_one(
            {"topic_id": topic.topic_id},
            doc,
            upsert=True,
        )
        return topic.topic_id

    async def get_topic(self, topic_id: str) -> TopicDTO | None:
        """Get a topic by ID."""
        doc = await self._client.topics.find_one({"topic_id": topic_id})
        return self._doc_to_topic(doc) if doc else None

    async def update_topic(self, topic: TopicDTO) -> None:
        """Update an existing topic."""
        await self.save_topic(topic)

    async def find_similar_topics(
        self,
        embedding: list[float],
        threshold: float,
    ) -> list[tuple[TopicDTO, float]]:
        """Find topics with similar embeddings.

        Uses MongoDB Atlas Vector Search when available for efficient
        approximate nearest neighbor (ANN) search. Falls back to
        in-memory cosine similarity calculation for non-Atlas deployments.

        Args:
            embedding: Query embedding vector
            threshold: Minimum similarity threshold (0.0 to 1.0)

        Returns:
            List of (TopicDTO, similarity) tuples, sorted by similarity desc
        """
        if not embedding:
            return []

        if self._vector_search_enabled:
            return await self._find_similar_topics_vector_search(embedding, threshold)
        return await self._find_similar_topics_fallback(embedding, threshold)

    async def _find_similar_topics_vector_search(
        self,
        embedding: list[float],
        threshold: float,
    ) -> list[tuple[TopicDTO, float]]:
        """Find similar topics using MongoDB Atlas Vector Search.

        Uses $vectorSearch aggregation for efficient ANN search.
        """
        results: list[tuple[TopicDTO, float]] = []

        # Build vector search aggregation pipeline
        pipeline = [
            {
                "$vectorSearch": {
                    "index": self._vector_search_index_name,
                    "path": "centroid_embedding",
                    "queryVector": embedding,
                    "numCandidates": self._vector_search_num_candidates,
                    "limit": self._vector_search_num_candidates,
                }
            },
            {
                "$addFields": {
                    "similarity_score": {"$meta": "vectorSearchScore"}
                }
            },
            {
                "$match": {
                    "similarity_score": {"$gte": threshold}
                }
            },
            {
                "$sort": {"similarity_score": -1}
            },
        ]

        try:
            cursor = self._client.topics.aggregate(pipeline)
            async for doc in cursor:
                topic = self._doc_to_topic(doc)
                similarity = doc.get("similarity_score", 0.0)
                results.append((topic, similarity))

            logger.debug(
                "vector_search_completed",
                num_results=len(results),
                threshold=threshold,
            )
            return results

        except Exception as e:
            # If vector search fails (index not ready, etc.), fall back
            logger.warning(
                "vector_search_failed_falling_back",
                error=str(e),
            )
            return await self._find_similar_topics_fallback(embedding, threshold)

    async def _find_similar_topics_fallback(
        self,
        embedding: list[float],
        threshold: float,
    ) -> list[tuple[TopicDTO, float]]:
        """Find similar topics using in-memory cosine similarity.

        Fallback implementation for non-Atlas deployments.
        """
        results: list[tuple[TopicDTO, float]] = []
        query_vec = np.array(embedding)
        query_norm = np.linalg.norm(query_vec)

        if query_norm == 0:
            return results

        # Fetch all topics with embeddings
        cursor = self._client.topics.find(
            {"centroid_embedding": {"$exists": True, "$ne": []}}
        )

        async for doc in cursor:
            topic = self._doc_to_topic(doc)
            if not topic.centroid_embedding:
                continue

            # Calculate cosine similarity
            doc_vec = np.array(topic.centroid_embedding)
            doc_norm = np.linalg.norm(doc_vec)
            if doc_norm == 0:
                continue

            similarity = float(np.dot(query_vec, doc_vec) / (query_norm * doc_norm))

            if similarity >= threshold:
                results.append((topic, similarity))

        # Sort by similarity descending
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    async def get_all_topics(self, limit: int = 1000) -> list[TopicDTO]:
        """Get all topics."""
        cursor = self._client.topics.find().limit(limit)
        return [self._doc_to_topic(doc) async for doc in cursor]

    # Evidence operations
    async def append_evidence(self, evidence: TopicEvidenceDTO) -> str:
        """Append evidence record (never update)."""
        doc = self._evidence_to_doc(evidence)
        await self._client.evidence.insert_one(doc)
        return evidence.evidence_id

    async def get_evidence_for_topic(self, topic_id: str) -> list[TopicEvidenceDTO]:
        """Get all evidence for a topic."""
        cursor = self._client.evidence.find({"topic_id": topic_id}).sort("timestamp", 1)
        return [self._doc_to_evidence(doc) async for doc in cursor]

    # Recall state operations
    async def save_recall_state(self, state: TopicRecallStateDTO) -> None:
        """Save or update recall state."""
        doc = self._recall_state_to_doc(state)
        await self._client.recall_states.replace_one(
            {"topic_id": state.topic_id},
            doc,
            upsert=True,
        )

    async def get_recall_state(self, topic_id: str) -> TopicRecallStateDTO | None:
        """Get recall state for a topic."""
        doc = await self._client.recall_states.find_one({"topic_id": topic_id})
        return self._doc_to_recall_state(doc) if doc else None

    async def get_due_topics(self, threshold: float) -> list[TopicRecallStateDTO]:
        """Get topics due for recall (strength below threshold)."""
        cursor = self._client.recall_states.find({"strength": {"$lt": threshold}}).sort(
            "strength", 1
        )
        return [self._doc_to_recall_state(doc) async for doc in cursor]

    async def get_all_recall_states(self) -> list[TopicRecallStateDTO]:
        """Get all recall states."""
        cursor = self._client.recall_states.find()
        return [self._doc_to_recall_state(doc) async for doc in cursor]

    # Document conversion helpers
    @staticmethod
    def _chat_to_doc(chat: ChatDTO) -> dict[str, Any]:
        return {
            "id": chat.id,
            "title": chat.title,
            "source": chat.source,
            "category": chat.category.value,
            "tags": chat.tags,
            "created_on": chat.created_on,
            "schema_version": chat.schema_version,
        }

    @staticmethod
    def _doc_to_chat(doc: dict[str, Any]) -> ChatDTO:
        return ChatDTO(
            id=doc["id"],
            title=doc["title"],
            source=doc["source"],
            category=ChatCategory(doc["category"]),
            tags=doc.get("tags", []),
            created_on=doc["created_on"],
            schema_version=doc.get("schema_version", 1),
        )

    @staticmethod
    def _message_to_doc(message: MessageDTO) -> dict[str, Any]:
        return {
            "message_id": message.message_id,
            "chat_id": message.chat_id,
            "user_content": message.user_content,
            "assistant_content": message.assistant_content,
            "created_on": message.created_on,
            "schema_version": message.schema_version,
        }

    @staticmethod
    def _doc_to_message(doc: dict[str, Any]) -> MessageDTO:
        return MessageDTO(
            message_id=doc["message_id"],
            chat_id=doc["chat_id"],
            user_content=doc.get("user_content", ""),
            assistant_content=doc.get("assistant_content", ""),
            created_on=doc["created_on"],
            schema_version=doc.get("schema_version", 1),
        )

    @staticmethod
    def _topic_to_doc(topic: TopicDTO) -> dict[str, Any]:
        return {
            "topic_id": topic.topic_id,
            "canonical_name": topic.canonical_name,
            "centroid_embedding": topic.centroid_embedding,
            "evidence_count": topic.evidence_count,
            "importance": topic.importance,
            "recall_strength": topic.recall_strength,
            "schema_version": topic.schema_version,
        }

    @staticmethod
    def _doc_to_topic(doc: dict[str, Any]) -> TopicDTO:
        return TopicDTO(
            topic_id=doc["topic_id"],
            canonical_name=doc["canonical_name"],
            centroid_embedding=doc.get("centroid_embedding", []),
            evidence_count=doc.get("evidence_count", 0),
            importance=doc.get("importance", 0.0),
            recall_strength=doc.get("recall_strength", 0.0),
            schema_version=doc.get("schema_version", 1),
        )

    @staticmethod
    def _evidence_to_doc(evidence: TopicEvidenceDTO) -> dict[str, Any]:
        return {
            "evidence_id": evidence.evidence_id,
            "topic_id": evidence.topic_id,
            "extracted_name": evidence.extracted_name,
            "source_message_id": evidence.source_message_id,
            "confidence": evidence.confidence,
            "timestamp": evidence.timestamp,
            "schema_version": evidence.schema_version,
        }

    @staticmethod
    def _doc_to_evidence(doc: dict[str, Any]) -> TopicEvidenceDTO:
        return TopicEvidenceDTO(
            evidence_id=doc["evidence_id"],
            topic_id=doc["topic_id"],
            extracted_name=doc["extracted_name"],
            source_message_id=doc["source_message_id"],
            confidence=doc["confidence"],
            timestamp=doc["timestamp"],
            schema_version=doc.get("schema_version", 1),
        )

    @staticmethod
    def _recall_state_to_doc(state: TopicRecallStateDTO) -> dict[str, Any]:
        return {
            "topic_id": state.topic_id,
            "strength": state.strength,
            "last_seen": state.last_seen,
            "last_updated": state.last_updated,
            "stability": state.stability,
            "schema_version": state.schema_version,
        }

    @staticmethod
    def _doc_to_recall_state(doc: dict[str, Any]) -> TopicRecallStateDTO:
        return TopicRecallStateDTO(
            topic_id=doc["topic_id"],
            strength=doc["strength"],
            last_seen=doc["last_seen"],
            last_updated=doc["last_updated"],
            stability=doc.get("stability", 1.0),
            schema_version=doc.get("schema_version", 1),
        )
