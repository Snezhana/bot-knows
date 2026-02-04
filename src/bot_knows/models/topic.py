"""Topic models for bot_knows.

These models represent canonical topics and their evidence
in the knowledge base.
"""

from pydantic import BaseModel, Field

__all__ = [
    "TopicDTO",
    "TopicEvidenceDTO",
]


class TopicDTO(BaseModel, frozen=True):
    """Canonical semantic topic.

    Topics are deduplicated semantic concepts extracted from messages.
    Each topic has a running centroid embedding for similarity matching.

    Attributes:
        topic_id: Deterministic topic ID
        canonical_name: Canonical/normalized topic name
        centroid_embedding: Running centroid of all evidence embeddings
        evidence_count: Number of evidence records (for centroid updates)
        importance: Topic importance score (0.0 - 1.0)
        recall_strength: Current recall strength (0.0 - 1.0)
        schema_version: Schema version for forward compatibility
    """

    topic_id: str = Field(description="Hash-based topic ID")
    canonical_name: str = Field(description="Canonical topic name")
    centroid_embedding: list[float] = Field(
        default_factory=list, description="Running centroid embedding"
    )
    evidence_count: int = Field(default=0, description="Number of evidence records")
    importance: float = Field(default=0.0, ge=0.0, le=1.0)
    recall_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    schema_version: int = Field(default=1)

    def with_updated_centroid(
        self,
        new_embedding: list[float],
    ) -> "TopicDTO":
        """Create a new TopicDTO with updated centroid embedding.

        Uses incremental centroid update formula:
            new_centroid = (old_centroid * n + new_embedding) / (n + 1)

        Args:
            new_embedding: New embedding to incorporate

        Returns:
            New TopicDTO with updated centroid and evidence_count
        """
        n = self.evidence_count
        if n == 0:
            # First embedding becomes the centroid
            new_centroid = new_embedding
        else:
            # Incremental update
            new_centroid = [
                (old * n + new) / (n + 1)
                for old, new in zip(self.centroid_embedding, new_embedding, strict=False)
            ]

        return TopicDTO(
            topic_id=self.topic_id,
            canonical_name=self.canonical_name,
            centroid_embedding=new_centroid,
            evidence_count=n + 1,
            importance=self.importance,
            recall_strength=self.recall_strength,
            schema_version=self.schema_version,
        )


class TopicEvidenceDTO(BaseModel, frozen=True):
    """Append-only evidence linking extraction to topic.

    Evidence records are never modified or deleted. They provide
    a complete audit trail of topic extractions.

    Attributes:
        evidence_id: Deterministic evidence ID
        topic_id: Parent topic ID
        extracted_name: Raw extracted topic name (before normalization)
        source_message_id: ID of the message this was extracted from
        confidence: Extraction confidence score (0.0 - 1.0)
        timestamp: Extraction timestamp in epoch seconds
        schema_version: Schema version for forward compatibility
    """

    evidence_id: str = Field(description="Hash-based evidence ID")
    topic_id: str = Field(description="Parent topic ID")
    extracted_name: str = Field(description="Raw extracted name")
    source_message_id: str = Field(description="Source message ID")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence")
    timestamp: int = Field(description="Epoch seconds")
    schema_version: int = Field(default=1)
