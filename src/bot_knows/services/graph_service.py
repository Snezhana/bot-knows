"""Graph service for bot_knows.

This module provides the service for managing the knowledge graph.
"""

from bot_knows.interfaces.graph import GraphServiceInterface
from bot_knows.logging import get_logger
from bot_knows.models.chat import ChatDTO
from bot_knows.models.message import MessageDTO
from bot_knows.models.topic import TopicDTO, TopicEvidenceDTO

__all__ = [
    "GraphService",
]

logger = get_logger(__name__)


class GraphService:
    """Service for managing the knowledge graph.

    Wraps the graph interface to provide higher-level operations
    for building and querying the knowledge graph.

    Graph structure:
    - (Message)-[:IS_PART_OF]->(Chat)
    - (Message)-[:FOLLOWS_AFTER]->(Message)
    - (Topic)-[:IS_SUPPORTED_BY {evidence}]->(Message)
    - (Topic)-[:RELATES_TO {similarity}]->(Topic)

    Example:
        service = GraphService(graph_interface)
        await service.add_chat_with_messages(chat, messages)
    """

    def __init__(self, graph: GraphServiceInterface) -> None:
        """Initialize service with graph interface.

        Args:
            graph: Graph interface implementation
        """
        self._graph = graph

    async def add_chat_with_messages(
        self,
        chat: ChatDTO,
        messages: list[MessageDTO],
    ) -> None:
        """Add a chat and its messages to the graph.

        Creates:
        - Chat node
        - Message nodes
        - IS_PART_OF edges (Message -> Chat)
        - FOLLOWS_AFTER edges (Message -> Message)

        Args:
            chat: Chat to add
            messages: Messages to add (should be ordered)
        """
        # Create chat node
        await self._graph.create_chat_node(chat)

        # Create message nodes and edges
        previous_message_id: str | None = None

        for message in messages:
            # Create message node
            await self._graph.create_message_node(message)

            # Create IS_PART_OF edge
            await self._graph.create_is_part_of_edge(
                message_id=message.message_id,
                chat_id=chat.id,
            )

            # Create FOLLOWS_AFTER edge if not first message
            if previous_message_id:
                await self._graph.create_follows_after_edge(
                    message_id=message.message_id,
                    previous_message_id=previous_message_id,
                )

            previous_message_id = message.message_id

        logger.debug(
            "chat_added_to_graph",
            chat_id=chat.id,
            message_count=len(messages),
        )

    async def add_topic_with_evidence(
        self,
        topic: TopicDTO,
        evidence: TopicEvidenceDTO,
    ) -> None:
        """Add a topic and link it to supporting message.

        Creates:
        - Topic node
        - IS_SUPPORTED_BY edge with evidence properties

        Args:
            topic: Topic to add
            evidence: Evidence linking topic to message
        """
        # Create topic node
        await self._graph.create_topic_node(topic)

        # Create IS_SUPPORTED_BY edge with evidence
        await self._graph.create_is_supported_by_edge(
            topic_id=topic.topic_id,
            message_id=evidence.source_message_id,
            evidence=evidence,
        )

        logger.debug(
            "topic_added_to_graph",
            topic_id=topic.topic_id,
            message_id=evidence.source_message_id,
        )

    async def add_evidence_to_existing_topic(
        self,
        topic: TopicDTO,
        evidence: TopicEvidenceDTO,
    ) -> None:
        """Add evidence to an existing topic.

        Creates:
        - IS_SUPPORTED_BY edge with evidence properties
        - Updates topic node properties

        Args:
            topic: Updated topic (with new centroid)
            evidence: New evidence
        """
        # Update topic node
        await self._graph.update_topic_node(topic)

        # Create IS_SUPPORTED_BY edge with evidence
        await self._graph.create_is_supported_by_edge(
            topic_id=topic.topic_id,
            message_id=evidence.source_message_id,
            evidence=evidence,
        )

        logger.debug(
            "evidence_added_to_topic",
            topic_id=topic.topic_id,
            evidence_id=evidence.evidence_id,
        )

    async def create_topic_relation(
        self,
        topic_id: str,
        related_topic_id: str,
        similatiry: float,
    ) -> None:
        """Create RELATES_TO edge between topics.

        Args:
            topic_id: Source topic ID
            related_topic_id: Related topic ID
            similatiry: Topics similatiry (0.0-1.0)
        """
        await self._graph.create_relates_to_edge(
            topic_id=topic_id,
            related_topic_id=related_topic_id,
            similarity=similatiry,
        )

    async def get_related_topics(
        self,
        topic_id: str,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Get topics related to a given topic.

        Args:
            topic_id: Topic to find relations for
            limit: Maximum results

        Returns:
            List of (topic_id, similarity) tuples
        """
        return await self._graph.get_related_topics(topic_id, limit)
