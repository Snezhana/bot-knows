"""Storage interface for bot_knows.

This module defines the Protocol for persistent storage operations.
"""

from typing import ClassVar, Protocol, runtime_checkable

from bot_knows.models.chat import ChatDTO
from bot_knows.models.message import MessageDTO
from bot_knows.models.recall import TopicRecallStateDTO
from bot_knows.models.topic import TopicDTO, TopicEvidenceDTO

__all__ = [
    "StorageInterface",
]


@runtime_checkable
class StorageInterface(Protocol):
    """Contract for persistent storage operations.

    Implementations should provide CRUD operations for
    chats, messages, topics, evidence, and recall state.
    """

    config_class: ClassVar[type | None] = None

    # Chat operations
    async def save_chat(self, chat: ChatDTO) -> str:
        """Save a chat to storage.

        Args:
            chat: Chat data to save

        Returns:
            Chat ID
        """
        ...

    async def get_chat(self, chat_id: str) -> ChatDTO | None:
        """Get a chat by ID.

        Args:
            chat_id: Chat ID to retrieve

        Returns:
            ChatDTO if found, None otherwise
        """
        ...

    async def chat_exists(self, chat_id: str) -> bool:
        """Check if a chat exists.

        Args:
            chat_id: Chat ID to check

        Returns:
            True if exists, False otherwise
        """
        ...

    async def find_chats_by_source(self, source: str) -> list[ChatDTO]:
        """Find all chats from a given source.

        Args:
            source: Import source to filter by

        Returns:
            List of matching chats
        """
        ...

    # Message operations
    async def save_message(self, message: MessageDTO) -> str:
        """Save a message to storage.

        Args:
            message: Message data to save

        Returns:
            Message ID
        """
        ...

    async def get_message(self, message_id: str) -> MessageDTO | None:
        """Get a message by ID.

        Args:
            message_id: Message ID to retrieve

        Returns:
            MessageDTO if found, None otherwise
        """
        ...

    async def get_messages_for_chat(self, chat_id: str) -> list[MessageDTO]:
        """Get all messages for a chat.

        Args:
            chat_id: Chat ID to query

        Returns:
            List of messages, ordered by timestamp
        """
        ...

    # Topic operations
    async def save_topic(self, topic: TopicDTO) -> str:
        """Save a topic to storage.

        Args:
            topic: Topic data to save

        Returns:
            Topic ID
        """
        ...

    async def get_topic(self, topic_id: str) -> TopicDTO | None:
        """Get a topic by ID.

        Args:
            topic_id: Topic ID to retrieve

        Returns:
            TopicDTO if found, None otherwise
        """
        ...

    async def update_topic(self, topic: TopicDTO) -> None:
        """Update an existing topic.

        Args:
            topic: Updated topic data
        """
        ...

    async def find_similar_topics(
        self,
        embedding: list[float],
        threshold: float,
    ) -> list[tuple[TopicDTO, float]]:
        """Find topics with similar embeddings.

        Args:
            embedding: Query embedding vector
            threshold: Minimum similarity threshold

        Returns:
            List of (TopicDTO, similarity) tuples, sorted by similarity desc
        """
        ...

    async def get_all_topics(self, limit: int = 1000) -> list[TopicDTO]:
        """Get all topics (for batch operations).

        Args:
            limit: Maximum number of topics to return

        Returns:
            List of topics
        """
        ...

    # Evidence operations
    async def append_evidence(self, evidence: TopicEvidenceDTO) -> str:
        """Append evidence record (never update or delete).

        Args:
            evidence: Evidence data to append

        Returns:
            Evidence ID
        """
        ...

    async def get_evidence_for_topic(self, topic_id: str) -> list[TopicEvidenceDTO]:
        """Get all evidence for a topic.

        Args:
            topic_id: Topic ID to query

        Returns:
            List of evidence records
        """
        ...

    # Recall state operations
    async def save_recall_state(self, state: TopicRecallStateDTO) -> None:
        """Save or update recall state for a topic.

        Args:
            state: Recall state to save
        """
        ...

    async def get_recall_state(self, topic_id: str) -> TopicRecallStateDTO | None:
        """Get recall state for a topic.

        Args:
            topic_id: Topic ID to query

        Returns:
            TopicRecallStateDTO if found, None otherwise
        """
        ...

    async def get_due_topics(self, threshold: float) -> list[TopicRecallStateDTO]:
        """Get topics due for recall review.

        Args:
            threshold: Strength threshold (topics below this are due)

        Returns:
            List of recall states for due topics
        """
        ...

    async def get_all_recall_states(self) -> list[TopicRecallStateDTO]:
        """Get all recall states (for batch decay updates).

        Returns:
            List of all recall states
        """
        ...
