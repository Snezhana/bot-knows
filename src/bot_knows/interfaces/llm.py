"""LLM interface for bot_knows.

This module defines the Protocol for LLM interactions
including chat classification and topic extraction.
"""

from typing import Protocol, runtime_checkable

from bot_knows.models.chat import ChatCategory

__all__ = [
    "LLMInterface",
]


@runtime_checkable
class LLMInterface(Protocol):
    """Contract for LLM interactions.

    Implementations should provide methods for classifying chats
    and extracting topics from messages.
    """

    async def classify_chat(
        self,
        first_pair: tuple[str, str],
        last_pair: tuple[str, str],
    ) -> tuple[ChatCategory, list[str]]:
        """Classify a chat and extract tags.

        Uses the first and last user-assistant pairs to determine
        the chat's category and relevant tags.

        Args:
            first_pair: (user_message, assistant_message) from start of chat
            last_pair: (user_message, assistant_message) from end of chat

        Returns:
            Tuple of (ChatCategory, list of tags)
        """
        ...

    async def extract_topics(
        self,
        user_content: str,
        assistant_content: str,
    ) -> list[tuple[str, float]]:
        """Extract topic candidates from a message pair.

        Args:
            user_content: User's message content
            assistant_content: Assistant's response content

        Returns:
            List of (topic_name, confidence) tuples
        """
        ...

    async def normalize_topic_name(self, extracted_name: str) -> str:
        """Normalize a topic name to canonical form.

        Args:
            extracted_name: Raw extracted topic name

        Returns:
            Normalized canonical topic name
        """
        ...
