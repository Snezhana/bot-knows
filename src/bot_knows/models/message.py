"""Message models for bot_knows.

These models represent processed messages in the knowledge base.
"""

from pydantic import BaseModel, Field

__all__ = [
    "MessageDTO",
]


class MessageDTO(BaseModel, frozen=True):
    """User-Assistant message pair.

    Messages are stored as user-assistant pairs rather than individual
    messages. This reflects the conversational nature of chat data
    and simplifies topic extraction.

    Attributes:
        message_id: Deterministic message ID (hash-based)
        chat_id: Parent chat ID
        user_content: User's message content (may be empty)
        assistant_content: Assistant's response content (may be empty)
        created_on: Message timestamp in epoch seconds
        schema_version: Schema version for forward compatibility
    """

    message_id: str = Field(description="Hash-based message ID")
    chat_id: str = Field(description="Parent chat ID")
    user_content: str = Field(default="", description="User's message")
    assistant_content: str = Field(default="", description="Assistant's response")
    created_on: int = Field(description="Epoch seconds")
    schema_version: int = Field(default=1)

    @property
    def combined_content(self) -> str:
        """Get combined user and assistant content for processing."""
        parts = []
        if self.user_content:
            parts.append(f"User: {self.user_content}")
        if self.assistant_content:
            parts.append(f"Assistant: {self.assistant_content}")
        return "\n\n".join(parts)

    @property
    def is_empty(self) -> bool:
        """Check if both user and assistant content are empty."""
        return not self.user_content and not self.assistant_content
