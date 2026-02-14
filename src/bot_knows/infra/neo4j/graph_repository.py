"""Neo4j graph repository for bot_knows.

This module provides the graph repository implementation for Neo4j.
"""

from typing import Any, Self

from bot_knows.config import Neo4jSettings
from bot_knows.infra.neo4j.client import Neo4jClient
from bot_knows.interfaces.graph import GraphServiceInterface
from bot_knows.logging import get_logger
from bot_knows.models.chat import ChatDTO
from bot_knows.models.message import MessageDTO
from bot_knows.models.topic import TopicDTO, TopicEvidenceDTO

__all__ = [
    "Neo4jGraphRepository",
]

logger = get_logger(__name__)


class Neo4jGraphRepository(GraphServiceInterface):
    """Neo4j implementation of GraphServiceInterface.

    Provides graph operations for the knowledge base including
    node creation, edge creation, and graph queries.
    """

    config_class = Neo4jSettings

    def __init__(self, client: Neo4jClient) -> None:
        """Initialize repository with Neo4j client.

        Args:
            client: Connected Neo4jClient instance
        """
        self._client = client
        self._owns_client = False

    @classmethod
    async def from_config(cls, config: Neo4jSettings) -> Self:
        """Factory method for BotKnows instantiation.

        Creates a Neo4jClient, connects, creates indexes/constraints, and returns repository.

        Args:
            config: Neo4j settings

        Returns:
            Connected Neo4jGraphRepository instance
        """
        client = Neo4jClient(config)
        await client.connect()
        await client.create_indexes()
        await client.create_constraints()
        instance = cls(client)
        instance._owns_client = True
        return instance

    @classmethod
    async def from_dict(cls, config: dict[str, Any]) -> Self:
        """Factory method for custom config dict.

        Args:
            config: Dictionary with Neo4j settings

        Returns:
            Connected Neo4jGraphRepository instance
        """
        settings = Neo4jSettings(**config)
        return await cls.from_config(settings)

    async def close(self) -> None:
        """Close owned resources."""
        if self._owns_client and self._client:
            await self._client.disconnect()

    # Node operations
    async def create_message_node(self, message: MessageDTO, chat: ChatDTO) -> str:
        """Create or update a Message node with chat metadata."""
        query = """
        MERGE (m:Message {message_id: $message_id})
        SET m.chat_id = $chat_id,
            m.chat_title = $chat_title,
            m.source = $source,
            m.category = $category,
            m.tags = $tags,
            m.created_on = $created_on,
            m.user_content = $user_content,
            m.assistant_content = $assistant_content
        RETURN m.message_id as id
        """
        await self._client.execute_write(
            query,
            {
                "message_id": message.message_id,
                "chat_id": message.chat_id,
                "chat_title": chat.title,
                "source": chat.source,
                "category": chat.category.value,
                "tags": chat.tags,
                "created_on": message.created_on,
                "user_content": message.user_content,
                "assistant_content": message.assistant_content,
            },
        )
        return message.message_id

    async def create_topic_node(self, topic: TopicDTO) -> str:
        """Create or update a Topic node."""
        query = """
        MERGE (t:Topic {topic_id: $topic_id})
        SET t.canonical_name = $canonical_name,
            t.importance = $importance,
            t.recall_strength = $recall_strength
        RETURN t.topic_id as id
        """
        await self._client.execute_write(
            query,
            {
                "topic_id": topic.topic_id,
                "canonical_name": topic.canonical_name,
                "importance": topic.importance,
                "recall_strength": topic.recall_strength,
            },
        )
        return topic.topic_id

    async def update_topic_node(self, topic: TopicDTO) -> None:
        """Update an existing Topic node."""
        await self.create_topic_node(topic)

    # Edge operations
    async def create_follows_after_edge(
        self,
        message_id: str,
        previous_message_id: str,
    ) -> None:
        """Create FOLLOWS_AFTER edge: (Message)-[:FOLLOWS_AFTER]->(Message)."""
        query = """
        MATCH (m1:Message {message_id: $message_id})
        MATCH (m2:Message {message_id: $previous_message_id})
        MERGE (m1)-[:FOLLOWS_AFTER]->(m2)
        """
        await self._client.execute_write(
            query,
            {
                "message_id": message_id,
                "previous_message_id": previous_message_id,
            },
        )

    async def create_is_supported_by_edge(
        self,
        topic_id: str,
        message_id: str,
        evidence: TopicEvidenceDTO,
    ) -> None:
        """Create IS_SUPPORTED_BY edge with evidence properties.

        (Topic)-[:IS_SUPPORTED_BY {evidence data}]->(Message)
        """
        query = """
        MATCH (t:Topic {topic_id: $topic_id})
        MATCH (m:Message {message_id: $message_id})
        MERGE (t)-[r:IS_SUPPORTED_BY {evidence_id: $evidence_id}]->(m)
        SET r.extracted_name = $extracted_name,
            r.confidence = $confidence,
            r.timestamp = $timestamp
        """
        await self._client.execute_write(
            query,
            {
                "topic_id": topic_id,
                "message_id": message_id,
                "evidence_id": evidence.evidence_id,
                "extracted_name": evidence.extracted_name,
                "confidence": evidence.confidence,
                "timestamp": evidence.timestamp,
            },
        )

    async def create_relates_to_edge(
        self,
        topic_id: str,
        related_topic_id: str,
        similarity: str,
    ) -> None:
        """Create RELATES_TO edge between topics."""
        query = """
        MATCH (t1:Topic {topic_id: $topic_id})
        MATCH (t2:Topic {topic_id: $related_topic_id})
        MERGE (t1)-[r:RELATES_TO]->(t2)
        SET r.similarity = $similarity
        """
        await self._client.execute_write(
            query,
            {
                "topic_id": topic_id,
                "related_topic_id": related_topic_id,
                "similarity": similarity,
            },
        )

    # Query operations
    async def get_messages_for_chat(self, chat_id: str) -> list[MessageDTO]:
        """Get all messages in a chat, ordered by created_on."""
        query = """
        MATCH (m:Message {chat_id: $chat_id})
        RETURN m.message_id as message_id,
               m.chat_id as chat_id,
               m.chat_title as chat_title,
               m.source as source,
               m.category as category,
               m.tags as tags,
               m.created_on as created_on,
               m.user_content as user_content,
               m.assistant_content as assistant_content
        ORDER BY m.created_on
        """
        records = await self._client.execute_query(query, {"chat_id": chat_id})

        return [
            MessageDTO(
                message_id=r["message_id"],
                chat_id=r["chat_id"],
                created_on=r["created_on"],
                user_content=r["user_content"] or "",
                assistant_content=r["assistant_content"] or "",
            )
            for r in records
        ]

    async def get_related_topics(
        self,
        topic_id: str,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Get topics related to a given topic."""
        query = """
        MATCH (t1:Topic {topic_id: $topic_id})-[r:RELATES_TO]->(t2:Topic)
        RETURN t2.topic_id as topic_id, r.similarity as similarity
        ORDER BY r.similarity DESC
        LIMIT $limit
        """
        records = await self._client.execute_query(
            query,
            {"topic_id": topic_id, "limit": limit},
        )
        return [(r["topic_id"], r["similarity"]) for r in records]

    async def get_topic_evidence(self, topic_id: str) -> list[dict[str, Any]]:
        """Get all evidence for a topic from IS_SUPPORTED_BY edges."""
        query = """
        MATCH (t:Topic {topic_id: $topic_id})-[r:IS_SUPPORTED_BY]->(m:Message)
        RETURN r.evidence_id as evidence_id,
               r.extracted_name as extracted_name,
               r.confidence as confidence,
               r.timestamp as timestamp,
               m.message_id as source_message_id
        ORDER BY r.timestamp
        """
        records = await self._client.execute_query(query, {"topic_id": topic_id})
        return [
            {
                "evidence_id": r["evidence_id"],
                "topic_id": topic_id,
                "extracted_name": r["extracted_name"],
                "confidence": r["confidence"],
                "timestamp": r["timestamp"],
                "source_message_id": r["source_message_id"],
            }
            for r in records
        ]

    async def get_chat_topics(self, chat_id: str) -> list[str]:
        """Get all topic IDs associated with a chat's messages."""
        query = """
        MATCH (t:Topic)-[:IS_SUPPORTED_BY]->(m:Message {chat_id: $chat_id})
        RETURN DISTINCT t.topic_id as topic_id
        """
        records = await self._client.execute_query(query, {"chat_id": chat_id})
        return [r["topic_id"] for r in records]
