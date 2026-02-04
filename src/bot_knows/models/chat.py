"""Chat models for bot_knows.

These models represent processed chats in the knowledge base.
"""

from enum import StrEnum

from pydantic import BaseModel, Field

__all__ = [
    "ChatCategory",
    "ChatDTO",
]


class ChatCategory(StrEnum):
    """Categories for chat classification.

    Used by the LLM-based classifier to categorize chats
    based on their content and purpose.
    """

    CODING = "coding"
    RESEARCH = "research"
    WRITING = "writing"
    BRAINSTORMING = "brainstorming"
    DEBUGGING = "debugging"
    LEARNING = "learning"
    GENERAL = "general"
    OTHER = "other"


class ChatDTO(BaseModel, frozen=True):
    """Public Chat data transfer object.

    Represents a processed chat in the knowledge base.
    Chats contain metadata only - message content is stored separately.

    Attributes:
        id: Deterministic chat ID (SHA256 of title + source + timestamp)
        title: Chat title (resolved from import or first message)
        source: Import source identifier
        category: LLM-classified category
        tags: Free-form tags from classification
        created_on: Chat creation timestamp in epoch seconds
        schema_version: Schema version for forward compatibility
    """

    id: str = Field(description="SHA256 hash of title + source + timestamp")
    title: str
    source: str = Field(description="Import source (chatgpt, claude, etc.)")
    category: ChatCategory = Field(default=ChatCategory.GENERAL)
    tags: list[str] = Field(default_factory=list)
    created_on: int = Field(description="Epoch seconds")
    schema_version: int = Field(default=1)
