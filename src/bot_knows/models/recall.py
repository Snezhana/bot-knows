"""Recall models for bot_knows.

These models represent the recall/spaced repetition state
for topics in the knowledge base.
"""

from pydantic import BaseModel, Field

from bot_knows.models.topic import TopicDTO

__all__ = [
    "TopicRecallStateDTO",
    "RecallItemDTO",
]


class TopicRecallStateDTO(BaseModel, frozen=True):
    """Persisted recall state per topic.

    Tracks the spaced repetition state for a topic including
    strength decay and stability.

    Attributes:
        topic_id: Topic ID this state belongs to
        strength: Current recall strength (0.0 - 1.0)
        last_seen: Last time topic was accessed (epoch seconds)
        last_updated: Last time decay was applied (epoch seconds)
        stability: Decay rate factor (higher = slower decay)
        schema_version: Schema version for forward compatibility
    """

    topic_id: str = Field(description="Topic ID")
    strength: float = Field(default=0.0, ge=0.0, le=1.0, description="Recall strength")
    last_seen: int = Field(description="Epoch seconds")
    last_updated: int = Field(description="Epoch seconds")
    stability: float = Field(default=1.0, ge=0.0, description="Decay rate factor")
    schema_version: int = Field(default=1)


class RecallItemDTO(BaseModel, frozen=True):
    """Topic ready for recall/review.

    Represents a topic that is due for review along with
    its recall state and related topics.

    Attributes:
        topic: The topic to review
        recall_state: Current recall state
        due_score: Priority score for recall (higher = more due)
        related_topics: IDs of semantically related topics
        schema_version: Schema version for forward compatibility
    """

    topic: TopicDTO
    recall_state: TopicRecallStateDTO
    due_score: float = Field(ge=0.0, description="Priority for recall")
    related_topics: list[str] = Field(default_factory=list, description="Related topic IDs")
    schema_version: int = Field(default=1)
