"""Neo4j graph repository for bot_knows.

This module provides the graph repository implementation for Neo4j.
"""

from typing import Any

from bot_knows.infra.neo4j.client import Neo4jClient
from bot_knows.logging import get_logger
from bot_knows.models.chat import ChatDTO
from bot_knows.models.message import MessageDTO
from bot_knows.models.topic import TopicDTO, TopicEvidenceDTO

__all__ = [
    "Neo4jGraphRepository",
]

logger = get_logger(__name__)


class Neo4jGraphRepository:
    """Neo4j implementation of GraphServiceInterface.

    Provides graph operations for the knowledge base including
    node creation, edge creation, and graph queries.
    """

    def __init__(self, client: Neo4jClient) -> None:
        """Initialize repository with Neo4j client.

        Args:
            client: Connected Neo4jClient instance
        """
        self._client = client

    # Node operations
    async def create_chat_node(self, chat: ChatDTO) -> str:
        """Create or update a Chat node."""
        query = """
        MERGE (c:Chat {id: $id})
        SET c.title = $title,
            c.source = $source,
            c.category = $category,
            c.tags = $tags,
            c.created_on = $created_on
        RETURN c.id as id
        """
        await self._client.execute_write(
            query,
            {
                "id": chat.id,
                "title": chat.title,
                "source": chat.source,
                "category": chat.category.value,
                "tags": chat.tags,
                "created_on": chat.created_on,
            },
        )
        return chat.id

    async def create_message_node(self, message: MessageDTO) -> str:
        """Create or update a Message node."""
        query = """
        MERGE (m:Message {message_id: $message_id})
        SET m.chat_id = $chat_id,
            m.created_on = $created_on
        RETURN m.message_id as id
        """
        await self._client.execute_write(
            query,
            {
                "message_id": message.message_id,
                "chat_id": message.chat_id,
                "created_on": message.created_on,
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
    async def create_is_part_of_edge(self, message_id: str, chat_id: str) -> None:
        """Create IS_PART_OF edge: (Message)-[:IS_PART_OF]->(Chat)."""
        query = """
        MATCH (m:Message {message_id: $message_id})
        MATCH (c:Chat {id: $chat_id})
        MERGE (m)-[:IS_PART_OF]->(c)
        """
        await self._client.execute_write(
            query,
            {"message_id": message_id, "chat_id": chat_id},
        )

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

    async def create_potentially_duplicate_of_edge(
        self,
        topic_id: str,
        existing_topic_id: str,
        similarity: float,
    ) -> None:
        """Create POTENTIALLY_DUPLICATE_OF edge between topics."""
        query = """
        MATCH (t1:Topic {topic_id: $topic_id})
        MATCH (t2:Topic {topic_id: $existing_topic_id})
        MERGE (t1)-[r:POTENTIALLY_DUPLICATE_OF]->(t2)
        SET r.similarity = $similarity
        """
        await self._client.execute_write(
            query,
            {
                "topic_id": topic_id,
                "existing_topic_id": existing_topic_id,
                "similarity": similarity,
            },
        )

    async def create_relates_to_edge(
        self,
        topic_id: str,
        related_topic_id: str,
        relation_type: str,
        weight: float,
    ) -> None:
        """Create RELATES_TO edge between topics."""
        query = """
        MATCH (t1:Topic {topic_id: $topic_id})
        MATCH (t2:Topic {topic_id: $related_topic_id})
        MERGE (t1)-[r:RELATES_TO]->(t2)
        SET r.type = $relation_type,
            r.weight = $weight
        """
        await self._client.execute_write(
            query,
            {
                "topic_id": topic_id,
                "related_topic_id": related_topic_id,
                "relation_type": relation_type,
                "weight": weight,
            },
        )

    # Query operations
    async def get_messages_for_chat(self, chat_id: str) -> list[MessageDTO]:
        """Get all messages in a chat, ordered by FOLLOWS_AFTER."""
        # Get messages ordered by created_on since FOLLOWS_AFTER may not exist
        query = """
        MATCH (m:Message)-[:IS_PART_OF]->(c:Chat {id: $chat_id})
        RETURN m.message_id as message_id,
               m.chat_id as chat_id,
               m.created_on as created_on
        ORDER BY m.created_on
        """
        records = await self._client.execute_query(query, {"chat_id": chat_id})
        return [
            MessageDTO(
                message_id=r["message_id"],
                chat_id=r["chat_id"],
                created_on=r["created_on"],
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
        RETURN t2.topic_id as topic_id, r.weight as weight
        ORDER BY r.weight DESC
        LIMIT $limit
        """
        records = await self._client.execute_query(
            query,
            {"topic_id": topic_id, "limit": limit},
        )
        return [(r["topic_id"], r["weight"]) for r in records]

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
        MATCH (t:Topic)-[:IS_SUPPORTED_BY]->(m:Message)-[:IS_PART_OF]->(c:Chat {id: $chat_id})
        RETURN DISTINCT t.topic_id as topic_id
        """
        records = await self._client.execute_query(query, {"chat_id": chat_id})
        return [r["topic_id"] for r in records]
