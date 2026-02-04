"""Recall service for bot_knows.

This module provides the evidence-weighted recall service with
spaced repetition-style decay and reinforcement.
"""

import math
import time
from typing import Literal

from bot_knows.interfaces.graph import GraphServiceInterface
from bot_knows.interfaces.storage import StorageInterface
from bot_knows.logging import get_logger
from bot_knows.models.recall import RecallItemDTO, TopicRecallStateDTO

__all__ = [
    "CONTEXT_WEIGHTS",
    "RecallService",
]

logger = get_logger(__name__)

# Context weights for reinforcement
CONTEXT_WEIGHTS: dict[str, float] = {
    "passive": 0.2,  # Passive read (background access)
    "active": 0.6,  # Active query (user explicitly asked)
    "recall": 1.0,  # Recall prompt (spaced repetition review)
}


class RecallService:
    """Evidence-weighted recall service.

    Implements a spaced repetition-inspired recall system with:
    - Time-based decay: strength *= exp(-Δt / (stability * 86400))
    - Reinforcement: strength += confidence * novelty * context_weight
    - Stability growth: stability += k * confidence
    - Semantic reinforcement: boost related topics

    Example:
        service = RecallService(storage, graph)

        # Reinforce when topic is accessed
        state = await service.reinforce(topic_id, confidence=0.9, context="active")

        # Get topics due for review
        due_topics = await service.get_due_topics(threshold=0.3)
    """

    def __init__(
        self,
        storage: StorageInterface,
        graph: GraphServiceInterface,
        stability_k: float = 0.1,
        semantic_boost: float = 0.1,
    ) -> None:
        """Initialize service with dependencies.

        Args:
            storage: Storage interface for recall states
            graph: Graph interface for related topics
            stability_k: Factor for stability growth on reinforcement
            semantic_boost: Factor for boosting related topics
        """
        self._storage = storage
        self._graph = graph
        self._stability_k = stability_k
        self._semantic_boost = semantic_boost

    async def reinforce(
        self,
        topic_id: str,
        confidence: float,
        novelty_factor: float = 1.0,
        context: Literal["passive", "active", "recall"] = "passive",
    ) -> TopicRecallStateDTO:
        """Reinforce a topic's recall strength.

        Formulas:
            delta = confidence * novelty_factor * context_weight
            strength = min(1.0, strength + delta)
            stability += k * confidence

        Also boosts semantically related topics.

        Args:
            topic_id: Topic to reinforce
            confidence: Evidence confidence (0.0-1.0)
            novelty_factor: How novel this reinforcement is
            context: Interaction context (passive/active/recall)

        Returns:
            Updated TopicRecallStateDTO
        """
        now = int(time.time())

        # Get or create state
        state = await self._storage.get_recall_state(topic_id)
        if not state:
            state = TopicRecallStateDTO(
                topic_id=topic_id,
                strength=0.0,
                last_seen=now,
                last_updated=now,
                stability=1.0,
            )

        # Apply decay first
        state = self._apply_decay(state, now)

        # Calculate reinforcement
        context_weight = CONTEXT_WEIGHTS.get(context, 0.2)
        delta = confidence * novelty_factor * context_weight
        new_strength = min(1.0, state.strength + delta)
        new_stability = state.stability + self._stability_k * confidence

        # Create updated state
        new_state = TopicRecallStateDTO(
            topic_id=topic_id,
            strength=new_strength,
            last_seen=now,
            last_updated=now,
            stability=new_stability,
        )

        # Save state
        await self._storage.save_recall_state(new_state)

        # Boost related topics
        await self._boost_related_topics(topic_id, state.strength)

        logger.debug(
            "topic_reinforced",
            topic_id=topic_id,
            old_strength=state.strength,
            new_strength=new_strength,
            context=context,
        )

        return new_state

    async def apply_decay(
        self,
        topic_id: str,
        current_time: int | None = None,
    ) -> TopicRecallStateDTO | None:
        """Apply time-based decay to a topic.

        Formula: strength *= exp(-Δt / (stability * 86400))

        Args:
            topic_id: Topic to decay
            current_time: Current time (epoch seconds), default: now

        Returns:
            Updated state or None if topic has no state
        """
        state = await self._storage.get_recall_state(topic_id)
        if not state:
            return None

        now = current_time or int(time.time())
        new_state = self._apply_decay(state, now)

        await self._storage.save_recall_state(new_state)
        return new_state

    async def batch_decay_update(self) -> int:
        """Apply decay to all topics (scheduled task).

        Returns:
            Number of topics updated
        """
        now = int(time.time())
        states = await self._storage.get_all_recall_states()

        updated = 0
        for state in states:
            new_state = self._apply_decay(state, now)
            if new_state.strength != state.strength:
                await self._storage.save_recall_state(new_state)
                updated += 1

        logger.info("batch_decay_completed", updated_count=updated)
        return updated

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
            limit: Maximum number of topics

        Returns:
            List of RecallItemDTO sorted by priority
        """
        states = await self._storage.get_due_topics(threshold)

        items: list[RecallItemDTO] = []
        for state in states[:limit]:
            topic = await self._storage.get_topic(state.topic_id)
            if not topic:
                continue

            # Get related topics
            related = await self._graph.get_related_topics(state.topic_id, limit=5)

            # Calculate due score
            due_score = self._calculate_due_score(state)

            items.append(
                RecallItemDTO(
                    topic=topic,
                    recall_state=state,
                    due_score=due_score,
                    related_topics=[r[0] for r in related],
                )
            )

        # Sort by due_score descending
        items.sort(key=lambda x: x.due_score, reverse=True)
        return items

    def _apply_decay(
        self,
        state: TopicRecallStateDTO,
        current_time: int,
    ) -> TopicRecallStateDTO:
        """Apply time-based decay to state.

        Formula: strength *= exp(-Δt / (stability * 86400))
        """
        delta_t = current_time - state.last_updated
        if delta_t <= 0:
            return state

        # stability is in days, so multiply by seconds per day
        decay_factor = math.exp(-delta_t / (state.stability * 86400))
        new_strength = state.strength * decay_factor

        return TopicRecallStateDTO(
            topic_id=state.topic_id,
            strength=new_strength,
            last_seen=state.last_seen,
            last_updated=current_time,
            stability=state.stability,
        )

    async def _boost_related_topics(
        self,
        topic_id: str,
        source_strength: float,
    ) -> None:
        """Boost strength of related topics.

        Formula: strength += source_strength * edge_weight * semantic_boost
        """
        related = await self._graph.get_related_topics(topic_id)

        for related_id, edge_weight in related:
            state = await self._storage.get_recall_state(related_id)
            if not state:
                continue

            boost = source_strength * edge_weight * self._semantic_boost
            new_strength = min(1.0, state.strength + boost)

            if new_strength > state.strength:
                new_state = TopicRecallStateDTO(
                    topic_id=related_id,
                    strength=new_strength,
                    last_seen=state.last_seen,
                    last_updated=state.last_updated,
                    stability=state.stability,
                )
                await self._storage.save_recall_state(new_state)

    def _calculate_due_score(self, state: TopicRecallStateDTO) -> float:
        """Calculate priority score for recall review.

        Higher score = more urgent for review.
        Factors: lower strength, older last_seen
        """
        now = time.time()
        age_days = (now - state.last_seen) / 86400

        # Lower strength = higher priority
        # Older = higher priority
        return (1.0 - state.strength) * (1.0 + age_days * 0.1)
