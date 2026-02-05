"""Graph service interface for bot_knows.

This module defines the Protocol for graph database operations.
"""

from typing import Any, ClassVar, Protocol, runtime_checkable

from bot_knows.models.chat import ChatDTO
from bot_knows.models.message import MessageDTO
from bot_knows.models.topic import TopicDTO, TopicEvidenceDTO

__all__ = [
    "GraphServiceInterface",
]


@runtime_checkable
class GraphServiceInterface(Protocol):
    """Contract for graph database operations.

    Implementations should provide methods for creating nodes,
    edges, and querying the knowledge graph.
    """

    config_class: ClassVar[type | None] = None

    # Node operations
    async def create_chat_node(self, chat: ChatDTO) -> str:
        """Create a Chat node in the graph.

        Args:
            chat: Chat data to store

        Returns:
            Node ID
        """
        ...

    async def create_message_node(self, message: MessageDTO) -> str:
        """Create a Message node in the graph.

        Args:
            message: Message data to store

        Returns:
            Node ID
        """
        ...

    async def create_topic_node(self, topic: TopicDTO) -> str:
        """Create a Topic node in the graph.

        Args:
            topic: Topic data to store

        Returns:
            Node ID
        """
        ...

    async def update_topic_node(self, topic: TopicDTO) -> None:
        """Update an existing Topic node.

        Args:
            topic: Updated topic data
        """
        ...

    # Edge operations
    async def create_is_part_of_edge(self, message_id: str, chat_id: str) -> None:
        """Create IS_PART_OF edge: (Message)-[:IS_PART_OF]->(Chat).

        Args:
            message_id: Source message ID
            chat_id: Target chat ID
        """
        ...

    async def create_follows_after_edge(
        self,
        message_id: str,
        previous_message_id: str,
    ) -> None:
        """Create FOLLOWS_AFTER edge: (Message)-[:FOLLOWS_AFTER]->(Message).

        Args:
            message_id: Current message ID
            previous_message_id: Previous message ID
        """
        ...

    async def create_is_supported_by_edge(
        self,
        topic_id: str,
        message_id: str,
        evidence: TopicEvidenceDTO,
    ) -> None:
        """Create IS_SUPPORTED_BY edge with evidence properties.

        (Topic)-[:IS_SUPPORTED_BY {evidence properties}]->(Message)

        Args:
            topic_id: Topic ID
            message_id: Supporting message ID
            evidence: Evidence data to store as edge properties
        """
        ...

    async def create_relates_to_edge(
        self,
        topic_id: str,
        related_topic_id: str,
        similarity: str,
    ) -> None:
        """Create RELATES_TO edge between topics.

        Args:
            topic_id: Source topic ID
            related_topic_id: Related topic ID
            similarity: Topics similarity (0.0 - 1.0)
        """
        ...

    # Query operations
    async def get_messages_for_chat(self, chat_id: str) -> list[MessageDTO]:
        """Get all messages in a chat, ordered by FOLLOWS_AFTER.

        Args:
            chat_id: Chat ID to query

        Returns:
            List of messages in order
        """
        ...

    async def get_related_topics(
        self,
        topic_id: str,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Get topics related to a given topic.

        Args:
            topic_id: Topic to find relations for
            limit: Maximum number of results

        Returns:
            List of (topic_id, similarity) tuples
        """
        ...

    async def get_topic_evidence(
        self,
        topic_id: str,
    ) -> list[dict[str, Any]]:
        """Get all evidence for a topic from IS_SUPPORTED_BY edges.

        Args:
            topic_id: Topic ID

        Returns:
            List of evidence properties from edges
        """
        ...

    async def get_chat_topics(self, chat_id: str) -> list[str]:
        """Get all topic IDs associated with a chat's messages.

        Traverses: (Chat)<-[:IS_PART_OF]-(Message)<-[:IS_SUPPORTED_BY]-(Topic)

        Args:
            chat_id: Chat ID to query

        Returns:
            List of unique topic IDs
        """
        ...
