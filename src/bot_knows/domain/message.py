"""Internal Message entity for bot_knows.

This module contains the internal Message domain model.
"""

import time
from dataclasses import dataclass, field

from bot_knows.models.message import MessageDTO

__all__ = [
    "Message",
]


@dataclass
class Message:
    """Internal Message entity.

    This is a mutable internal representation used during processing.
    Convert to MessageDTO for persistence and external use.
    """

    message_id: str
    chat_id: str
    user_content: str = ""
    assistant_content: str = ""
    created_on: int = field(default_factory=lambda: int(time.time()))

    @property
    def combined_content(self) -> str:
        """Get combined user and assistant content."""
        parts = []
        if self.user_content:
            parts.append(f"User: {self.user_content}")
        if self.assistant_content:
            parts.append(f"Assistant: {self.assistant_content}")
        return "\n\n".join(parts)

    @property
    def is_empty(self) -> bool:
        """Check if both contents are empty."""
        return not self.user_content and not self.assistant_content

    def to_dto(self) -> MessageDTO:
        """Convert to immutable DTO for persistence."""
        return MessageDTO(
            message_id=self.message_id,
            chat_id=self.chat_id,
            user_content=self.user_content,
            assistant_content=self.assistant_content,
            created_on=self.created_on,
        )

    @classmethod
    def from_dto(cls, dto: MessageDTO) -> "Message":
        """Create from DTO."""
        return cls(
            message_id=dto.message_id,
            chat_id=dto.chat_id,
            user_content=dto.user_content,
            assistant_content=dto.assistant_content,
            created_on=dto.created_on,
        )
