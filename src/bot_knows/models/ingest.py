"""Ingestion boundary models for bot_knows.

These frozen Pydantic models define the contract between import adapters
and the domain processing layer. They are immutable and validated at creation.
"""

from typing import Literal

from pydantic import BaseModel, Field

__all__ = [
    "IngestMessage",
    "ChatIngest",
]


class IngestMessage(BaseModel, frozen=True):
    """Single message from import source.

    This is a frozen (immutable) model representing one message
    in its raw form from the import source.

    Attributes:
        role: Message author role (user, assistant, or system)
        content: Message text content
        timestamp: Message timestamp in epoch seconds
        chat_id: Provider's original chat/conversation identifier
        schema_version: Schema version for forward compatibility
    """

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: int = Field(description="Epoch seconds")
    chat_id: str = Field(description="Provider's chat identifier")
    schema_version: int = Field(default=1)


class ChatIngest(BaseModel, frozen=True):
    """Complete chat from import source.

    This is a frozen (immutable) model representing one complete chat
    conversation ready for domain processing.

    Attributes:
        source: Import source identifier (e.g., "chatgpt", "claude")
        imported_chat_timestamp: Chat creation/import timestamp in epoch seconds
        title: Chat title (may be None if not provided by source)
        messages: List of messages in the chat, ordered by timestamp
        provider: Original provider name (for provenance)
        conversation_id: Provider's original conversation ID
        schema_version: Schema version for forward compatibility
    """

    source: str = Field(description="Import source (chatgpt, claude, etc.)")
    imported_chat_timestamp: int = Field(description="Epoch seconds")
    title: str | None = Field(default=None)
    messages: list[IngestMessage] = Field(default_factory=list)
    provider: str | None = Field(default=None, description="Original provider")
    conversation_id: str | None = Field(default=None, description="Provider's conversation ID")
    schema_version: int = Field(default=1)

    @property
    def message_count(self) -> int:
        """Get the number of messages in this chat."""
        return len(self.messages)

    @property
    def has_messages(self) -> bool:
        """Check if this chat has any messages."""
        return len(self.messages) > 0
