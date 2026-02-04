"""Internal TopicEvidence entity for bot_knows.

This module contains the internal TopicEvidence domain model.
Evidence records are append-only and never modified.
"""

import time
from dataclasses import dataclass, field

from bot_knows.models.topic import TopicEvidenceDTO

__all__ = [
    "TopicEvidence",
]


@dataclass(frozen=True)
class TopicEvidence:
    """Internal TopicEvidence entity.

    Evidence records are append-only - they are never modified or deleted.
    This provides a complete audit trail of topic extractions.

    Note: This dataclass is frozen (immutable) to enforce append-only semantics.
    """

    evidence_id: str
    topic_id: str
    extracted_name: str
    source_message_id: str
    confidence: float
    timestamp: int = field(default_factory=lambda: int(time.time()))

    def to_dto(self) -> TopicEvidenceDTO:
        """Convert to immutable DTO for persistence."""
        return TopicEvidenceDTO(
            evidence_id=self.evidence_id,
            topic_id=self.topic_id,
            extracted_name=self.extracted_name,
            source_message_id=self.source_message_id,
            confidence=self.confidence,
            timestamp=self.timestamp,
        )

    @classmethod
    def from_dto(cls, dto: TopicEvidenceDTO) -> "TopicEvidence":
        """Create from DTO."""
        return cls(
            evidence_id=dto.evidence_id,
            topic_id=dto.topic_id,
            extracted_name=dto.extracted_name,
            source_message_id=dto.source_message_id,
            confidence=dto.confidence,
            timestamp=dto.timestamp,
        )
