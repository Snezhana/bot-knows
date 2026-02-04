"""Semantic deduplication service for bot_knows.

This module provides the service for deduplicating topics based on
semantic similarity using embedding vectors.
"""

from dataclasses import dataclass
from enum import StrEnum

from bot_knows.interfaces.embedding import EmbeddingServiceInterface
from bot_knows.interfaces.storage import StorageInterface
from bot_knows.logging import get_logger
from bot_knows.models.topic import TopicDTO

__all__ = [
    "DedupAction",
    "DedupResult",
    "DedupService",
]

logger = get_logger(__name__)


class DedupAction(StrEnum):
    """Actions resulting from deduplication check."""

    MERGE = "merge"
    """Similarity >= high_threshold: same topic, merge evidence"""

    SOFT_MATCH = "soft_match"
    """Similarity between low and high threshold: new topic + POTENTIALLY_DUPLICATE_OF edge"""

    NEW = "new"
    """Similarity < low_threshold: completely new topic"""


@dataclass
class DedupResult:
    """Result of deduplication check."""

    action: DedupAction
    existing_topic: TopicDTO | None = None
    similarity: float = 0.0


class DedupService:
    """Semantic deduplication service.

    Compares candidate topics against existing topics using
    embedding similarity to determine:
    - MERGE (>= 0.92): Same topic, link evidence to existing
    - SOFT_MATCH (0.80-0.92): Create new topic with POTENTIALLY_DUPLICATE_OF edge
    - NEW (< 0.80): Create completely new topic

    Example:
        service = DedupService(embedding_service, storage)
        result = await service.check_duplicate(embedding)
        if result.action == DedupAction.MERGE:
            # Link evidence to result.existing_topic
    """

    def __init__(
        self,
        embedding_service: EmbeddingServiceInterface,
        storage: StorageInterface,
        high_threshold: float = 0.92,
        low_threshold: float = 0.80,
    ) -> None:
        """Initialize service with dependencies.

        Args:
            embedding_service: Embedding service for similarity calculation
            storage: Storage interface for topic lookup
            high_threshold: Similarity threshold for MERGE (default: 0.92)
            low_threshold: Similarity threshold for SOFT_MATCH (default: 0.80)
        """
        self._embedding = embedding_service
        self._storage = storage
        self._high_threshold = high_threshold
        self._low_threshold = low_threshold

    async def check_duplicate(
        self,
        candidate_embedding: list[float],
    ) -> DedupResult:
        """Check if candidate embedding matches existing topics.

        Args:
            candidate_embedding: Embedding vector for candidate topic

        Returns:
            DedupResult with action and matched topic (if any)
        """
        # Find similar topics above low threshold
        similar_topics = await self._storage.find_similar_topics(
            embedding=candidate_embedding,
            threshold=self._low_threshold,
        )

        if not similar_topics:
            return DedupResult(action=DedupAction.NEW)

        # Get best match (highest similarity)
        best_topic, best_similarity = similar_topics[0]

        if best_similarity >= self._high_threshold:
            logger.debug(
                "dedup_merge",
                topic_id=best_topic.topic_id,
                similarity=best_similarity,
            )
            return DedupResult(
                action=DedupAction.MERGE,
                existing_topic=best_topic,
                similarity=best_similarity,
            )

        # Between thresholds: soft match
        logger.debug(
            "dedup_soft_match",
            topic_id=best_topic.topic_id,
            similarity=best_similarity,
        )
        return DedupResult(
            action=DedupAction.SOFT_MATCH,
            existing_topic=best_topic,
            similarity=best_similarity,
        )

    async def find_best_match(
        self,
        candidate_embedding: list[float],
        min_similarity: float = 0.5,
    ) -> tuple[TopicDTO, float] | None:
        """Find the best matching topic above minimum similarity.

        Args:
            candidate_embedding: Embedding vector to match
            min_similarity: Minimum similarity threshold

        Returns:
            (TopicDTO, similarity) tuple or None if no match
        """
        similar_topics = await self._storage.find_similar_topics(
            embedding=candidate_embedding,
            threshold=min_similarity,
        )

        if similar_topics:
            return similar_topics[0]
        return None

    @property
    def high_threshold(self) -> float:
        """Get high similarity threshold (MERGE)."""
        return self._high_threshold

    @property
    def low_threshold(self) -> float:
        """Get low similarity threshold (SOFT_MATCH)."""
        return self._low_threshold
