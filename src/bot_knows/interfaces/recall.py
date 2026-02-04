"""Recall service interface for bot_knows.

This module defines the Protocol for recall/spaced repetition operations.
"""

from typing import Literal, Protocol, runtime_checkable

from bot_knows.models.recall import RecallItemDTO, TopicRecallStateDTO

__all__ = [
    "RecallServiceInterface",
]


@runtime_checkable
class RecallServiceInterface(Protocol):
    """Contract for recall/spaced repetition operations.

    Implementations should provide methods for reinforcing topics,
    applying decay, and getting topics due for review.
    """

    async def reinforce(
        self,
        topic_id: str,
        confidence: float,
        novelty_factor: float = 1.0,
        context: Literal["passive", "active", "recall"] = "passive",
    ) -> TopicRecallStateDTO:
        """Reinforce a topic's recall strength.

        Formula:
            delta = confidence * novelty_factor * context_weight
            strength = min(1.0, strength + delta)
            stability += k * confidence

        Args:
            topic_id: Topic to reinforce
            confidence: Evidence confidence (0.0 - 1.0)
            novelty_factor: How novel this reinforcement is
            context: Reinforcement context (passive=0.2, active=0.6, recall=1.0)

        Returns:
            Updated recall state
        """
        ...

    async def apply_decay(
        self,
        topic_id: str,
        current_time: int | None = None,
    ) -> TopicRecallStateDTO:
        """Apply time-based decay to a topic.

        Formula:
            strength *= exp(-Δt / (stability * 86400))

        Args:
            topic_id: Topic to decay
            current_time: Current time in epoch seconds (default: now)

        Returns:
            Updated recall state
        """
        ...

    async def batch_decay_update(self) -> int:
        """Apply decay to all topics (scheduled task).

        Returns:
            Number of topics updated
        """
        ...

    async def get_due_topics(
        self,
        threshold: float = 0.3,
        limit: int = 10,
    ) -> list[RecallItemDTO]:
        """Get topics due for recall review.

        Topics with strength below threshold are considered due.
        Results are sorted by due_score (higher = more urgent).

        Args:
            threshold: Strength threshold for being "due"
            limit: Maximum number of topics to return

        Returns:
            List of RecallItemDTO sorted by priority
        """
        ...
