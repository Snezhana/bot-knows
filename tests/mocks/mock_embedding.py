"""Mock embedding service for testing."""

import hashlib

import numpy as np


class MockEmbeddingService:
    """Mock embedding service for testing.

    Generates deterministic embeddings based on text hash.
    """

    def __init__(self, dimensions: int = 1536) -> None:
        self._dimensions = dimensions

    async def embed(self, text: str) -> list[float]:
        """Generate deterministic embedding from text."""
        return self._generate_embedding(text)

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        return [self._generate_embedding(t) for t in texts]

    async def similarity(
        self,
        embedding1: list[float],
        embedding2: list[float],
    ) -> float:
        """Calculate cosine similarity."""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def _generate_embedding(self, text: str) -> list[float]:
        """Generate deterministic embedding from text hash."""
        # Use hash to seed random generator for determinism
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        seed = int(text_hash[:8], 16)
        rng = np.random.default_rng(seed)

        # Generate normalized embedding
        embedding = rng.standard_normal(self._dimensions)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm

        return embedding.tolist()
