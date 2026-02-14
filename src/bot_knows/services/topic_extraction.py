"""Topic extraction service for bot_knows.

This module provides the service for extracting topics from messages.
"""

import time

from bot_knows.interfaces.embedding import EmbeddingServiceInterface
from bot_knows.interfaces.llm import LLMInterface
from bot_knows.logging import get_logger
from bot_knows.models.message import MessageDTO
from bot_knows.models.topic import TopicDTO, TopicEvidenceDTO
from bot_knows.utils.hashing import generate_evidence_id, generate_topic_id

__all__ = [
    "TopicCandidate",
    "TopicExtractionService",
]

logger = get_logger(__name__)


class TopicCandidate:
    """Represents a candidate topic extracted from a message.

    This is an intermediate representation before deduplication.
    """

    def __init__(
        self,
        extracted_name: str,
        confidence: float,
        embedding: list[float],
        source_message_id: str,
    ) -> None:
        self.extracted_name = extracted_name
        self.confidence = confidence
        self.embedding = embedding
        self.source_message_id = source_message_id


class TopicExtractionService:
    """Service for extracting topics from messages.

    Uses LLM for topic extraction and embedding service for
    generating embeddings for semantic matching.

    Example:
        service = TopicExtractionService(llm, embedding_service)
        candidates = await service.extract(message)
    """

    def __init__(
        self,
        llm: LLMInterface,
        embedding_service: EmbeddingServiceInterface,
    ) -> None:
        """Initialize service with dependencies.

        Args:
            llm: LLM interface for topic extraction
            embedding_service: Embedding service for vector generation
        """
        self._llm = llm
        self._embedding = embedding_service

    async def extract(self, message: MessageDTO) -> list[TopicCandidate]:
        """Extract topic candidates from a message.

        Args:
            message: Message to extract topics from

        Returns:
            List of TopicCandidate objects
        """
        if message.is_empty:
            return []

        # Extract raw topics using LLM
        try:
            raw_topics = await self._llm.extract_topics(
                message.user_content,
                message.assistant_content,
            )
        except Exception as e:
            logger.warning("topic_extraction_failed", error=str(e))
            return []

        if not raw_topics:
            return []

        # Generate embeddings for all topics
        topic_names = [name.lower() for name, _ in raw_topics]
        try:
            embeddings = await self._embedding.embed_batch(topic_names)
        except Exception as e:
            logger.warning("embedding_generation_failed", error=str(e))
            return []

        # Create candidates
        candidates = []
        for (name, confidence), embedding in zip(raw_topics, embeddings, strict=False):
            candidate = TopicCandidate(
                extracted_name=name,
                confidence=confidence,
                embedding=embedding,
                source_message_id=message.message_id,
            )
            candidates.append(candidate)

        logger.debug(
            "topics_extracted",
            message_id=message.message_id,
            count=len(candidates),
        )

        return candidates

    async def create_topic_and_evidence(
        self,
        candidate: TopicCandidate,
        canonical_name: str | None = None,
    ) -> tuple[TopicDTO, TopicEvidenceDTO]:
        """Create new TopicDTO and TopicEvidenceDTO from candidate.

        Args:
            candidate: Topic candidate
            canonical_name: Normalized name (if None, uses extracted_name)

        Returns:
            Tuple of (TopicDTO, TopicEvidenceDTO)
        """
        name = canonical_name or await self._llm.normalize_topic_name(candidate.extracted_name)
        now = int(time.time())

        topic_id = generate_topic_id(name, candidate.source_message_id)
        evidence_id = generate_evidence_id(
            topic_id=topic_id,
            extracted_name=candidate.extracted_name,
            source_message_id=candidate.source_message_id,
            timestamp=now,
        )

        topic = TopicDTO(
            topic_id=topic_id,
            canonical_name=name,
            centroid_embedding=candidate.embedding,
            evidence_count=1,
            importance=candidate.confidence * 0.1,  # Initial importance
            recall_strength=0.0,
        )

        evidence = TopicEvidenceDTO(
            evidence_id=evidence_id,
            topic_id=topic_id,
            extracted_name=candidate.extracted_name,
            source_message_id=candidate.source_message_id,
            confidence=candidate.confidence,
            timestamp=now,
        )

        return topic, evidence

    async def create_evidence_for_existing(
        self,
        candidate: TopicCandidate,
        existing_topic: TopicDTO,
    ) -> tuple[TopicDTO, TopicEvidenceDTO]:
        """Create evidence for existing topic and update centroid.

        Args:
            candidate: Topic candidate
            existing_topic: Existing topic to link to

        Returns:
            Tuple of (updated TopicDTO, new TopicEvidenceDTO)
        """
        now = int(time.time())

        evidence_id = generate_evidence_id(
            topic_id=existing_topic.topic_id,
            extracted_name=candidate.extracted_name,
            source_message_id=candidate.source_message_id,
            timestamp=now,
        )

        evidence = TopicEvidenceDTO(
            evidence_id=evidence_id,
            topic_id=existing_topic.topic_id,
            extracted_name=candidate.extracted_name,
            source_message_id=candidate.source_message_id,
            confidence=candidate.confidence,
            timestamp=now,
        )

        # Update topic centroid
        updated_topic = existing_topic.with_updated_centroid(candidate.embedding)

        return updated_topic, evidence
