"""Embedding service interface for bot_knows.

This module defines the Protocol for embedding generation services.
"""

from typing import Protocol, runtime_checkable

__all__ = [
    "EmbeddingServiceInterface",
]


@runtime_checkable
class EmbeddingServiceInterface(Protocol):
    """Contract for embedding generation services.

    Implementations should provide methods for generating
    embeddings from text and computing similarity scores.
    """

    async def embed(self, text: str) -> list[float]:
        """Generate embedding vector for text.

        Args:
            text: Input text to embed

        Returns:
            Embedding vector as list of floats
        """
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        ...

    async def similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        """Compute cosine similarity between embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Similarity score between 0.0 and 1.0
        """
        ...
