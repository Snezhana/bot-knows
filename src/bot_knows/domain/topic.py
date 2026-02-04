"""Internal Topic entity for bot_knows.

This module contains the internal Topic domain model with
recall business logic including decay and reinforcement.
"""

import math
import time
from dataclasses import dataclass, field

from bot_knows.models.topic import TopicDTO

__all__ = [
    "Topic",
]


@dataclass
class Topic:
    """Internal Topic entity with recall business logic.

    This is a mutable internal representation used during processing.
    Includes methods for decay and reinforcement calculations.
    """

    topic_id: str
    canonical_name: str
    centroid_embedding: list[float] = field(default_factory=list)
    evidence_count: int = 0
    importance: float = 0.0
    recall_strength: float = 0.0
    stability: float = 1.0
    last_seen: int = field(default_factory=lambda: int(time.time()))
    last_updated: int = field(default_factory=lambda: int(time.time()))

    def update_centroid(self, new_embedding: list[float]) -> None:
        """Incrementally update centroid embedding.

        Uses formula: new_centroid = (old_centroid * n + new_embedding) / (n + 1)

        Args:
            new_embedding: New embedding to incorporate
        """
        n = self.evidence_count
        if n == 0:
            self.centroid_embedding = list(new_embedding)
        else:
            self.centroid_embedding = [
                (old * n + new) / (n + 1)
                for old, new in zip(self.centroid_embedding, new_embedding, strict=False)
            ]
        self.evidence_count += 1

    def reinforce(
        self,
        confidence: float,
        novelty_factor: float = 1.0,
        context_weight: float = 1.0,
        stability_k: float = 0.1,
    ) -> None:
        """Reinforce topic recall strength.

        Context weights:
            - passive: 0.2 (reading without interaction)
            - active: 0.6 (actively querying)
            - recall: 1.0 (responding to recall prompt)

        Formula:
            delta = confidence * novelty_factor * context_weight
            strength = min(1.0, strength + delta)
            stability += k * confidence

        Args:
            confidence: Evidence confidence (0.0 - 1.0)
            novelty_factor: How novel this reinforcement is
            context_weight: Weight based on interaction type
            stability_k: Stability increment factor
        """
        delta = confidence * novelty_factor * context_weight
        self.recall_strength = min(1.0, self.recall_strength + delta)
        self.stability += stability_k * confidence
        self.last_seen = int(time.time())
        self.last_updated = int(time.time())

    def apply_decay(self, current_time: int | None = None) -> None:
        """Apply time-based decay to recall strength.

        Formula: strength *= exp(-Δt / (stability * 86400))

        Higher stability means slower decay.

        Args:
            current_time: Current time in epoch seconds (default: now)
        """
        now = current_time or int(time.time())
        delta_t = now - self.last_updated
        if delta_t > 0:
            # Stability is multiplied by seconds per day for the decay rate
            decay_factor = math.exp(-delta_t / (self.stability * 86400))
            self.recall_strength *= decay_factor
            self.last_updated = now

    def increment_importance(self, delta: float = 0.1) -> None:
        """Increment importance score.

        Args:
            delta: Amount to increment (capped at 1.0)
        """
        self.importance = min(1.0, self.importance + delta)

    def to_dto(self) -> TopicDTO:
        """Convert to immutable DTO for persistence."""
        return TopicDTO(
            topic_id=self.topic_id,
            canonical_name=self.canonical_name,
            centroid_embedding=list(self.centroid_embedding),
            evidence_count=self.evidence_count,
            importance=self.importance,
            recall_strength=self.recall_strength,
        )

    @classmethod
    def from_dto(cls, dto: TopicDTO) -> "Topic":
        """Create from DTO."""
        return cls(
            topic_id=dto.topic_id,
            canonical_name=dto.canonical_name,
            centroid_embedding=list(dto.centroid_embedding),
            evidence_count=dto.evidence_count,
            importance=dto.importance,
            recall_strength=dto.recall_strength,
        )
