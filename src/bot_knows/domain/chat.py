"""Internal Chat entity for bot_knows.

This module contains the internal Chat domain model with business logic.
"""

from dataclasses import dataclass, field
import time

from bot_knows.models.chat import ChatCategory, ChatDTO

__all__ = [
    "Chat",
]


@dataclass
class Chat:
    """Internal Chat entity with business logic.

    This is a mutable internal representation used during processing.
    Convert to ChatDTO for persistence and external use.
    """

    id: str
    title: str
    source: str
    category: ChatCategory = ChatCategory.GENERAL
    tags: list[str] = field(default_factory=list)
    created_on: int = field(default_factory=lambda: int(time.time()))

    def add_tag(self, tag: str) -> None:
        """Add a tag if not already present."""
        if tag and tag not in self.tags:
            self.tags.append(tag)

    def add_tags(self, tags: list[str]) -> None:
        """Add multiple tags."""
        for tag in tags:
            self.add_tag(tag)

    def to_dto(self) -> ChatDTO:
        """Convert to immutable DTO for persistence."""
        return ChatDTO(
            id=self.id,
            title=self.title,
            source=self.source,
            category=self.category,
            tags=list(self.tags),
            created_on=self.created_on,
        )

    @classmethod
    def from_dto(cls, dto: ChatDTO) -> "Chat":
        """Create from DTO."""
        return cls(
            id=dto.id,
            title=dto.title,
            source=dto.source,
            category=dto.category,
            tags=list(dto.tags),
            created_on=dto.created_on,
        )
