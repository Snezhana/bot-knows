"""Redis cache implementations for bot_knows.

This module provides caching utilities for embeddings
and frequently accessed data.
"""

import hashlib
import json
from typing import Any

from bot_knows.infra.redis.client import RedisClient
from bot_knows.logging import get_logger

__all__ = [
    "EmbeddingCache",
    "TopicCache",
]

logger = get_logger(__name__)


class EmbeddingCache:
    """Redis cache for text embeddings.

    Caches embeddings to avoid redundant API calls.
    Falls back gracefully if Redis is unavailable.
    """

    def __init__(
        self,
        redis_client: RedisClient,
        ttl: int = 86400,  # 24 hours
        prefix: str = "emb:",
    ) -> None:
        """Initialize embedding cache.

        Args:
            redis_client: Redis client instance
            ttl: Cache TTL in seconds (default: 24 hours)
            prefix: Key prefix for embedding cache
        """
        self._redis = redis_client
        self._ttl = ttl
        self._prefix = prefix

    def _make_key(self, text: str) -> str:
        """Generate cache key for text."""
        text_hash = hashlib.sha256(text.encode()).hexdigest()
        return f"{self._prefix}{text_hash}"

    async def get(self, text: str) -> list[float] | None:
        """Get cached embedding for text.

        Args:
            text: Input text

        Returns:
            Embedding vector if cached, None otherwise
        """
        if not self._redis.is_connected:
            return None

        key = self._make_key(text)
        cached = await self._redis.get(key)

        if cached:
            try:
                return json.loads(cached)
            except json.JSONDecodeError:
                return None

        return None

    async def set(self, text: str, embedding: list[float]) -> bool:
        """Cache embedding for text.

        Args:
            text: Input text
            embedding: Embedding vector to cache

        Returns:
            True if cached successfully
        """
        if not self._redis.is_connected:
            return False

        key = self._make_key(text)
        return await self._redis.set(key, json.dumps(embedding), ex=self._ttl)

    async def get_or_compute(
        self,
        text: str,
        compute_fn: Any,
    ) -> list[float]:
        """Get cached embedding or compute and cache.

        Args:
            text: Input text
            compute_fn: Async function to compute embedding if not cached

        Returns:
            Embedding vector
        """
        # Try cache first
        cached = await self.get(text)
        if cached is not None:
            return cached

        # Compute embedding
        embedding = await compute_fn(text)

        # Cache result
        await self.set(text, embedding)

        return embedding


class TopicCache:
    """Redis cache for hot topics.

    Caches frequently accessed topic data to reduce
    database lookups.
    """

    def __init__(
        self,
        redis_client: RedisClient,
        ttl: int = 3600,  # 1 hour
        prefix: str = "topic:",
    ) -> None:
        """Initialize topic cache.

        Args:
            redis_client: Redis client instance
            ttl: Cache TTL in seconds (default: 1 hour)
            prefix: Key prefix for topic cache
        """
        self._redis = redis_client
        self._ttl = ttl
        self._prefix = prefix

    def _make_key(self, topic_id: str) -> str:
        """Generate cache key for topic."""
        return f"{self._prefix}{topic_id}"

    async def get(self, topic_id: str) -> dict[str, Any] | None:
        """Get cached topic data.

        Args:
            topic_id: Topic ID

        Returns:
            Topic data dict if cached, None otherwise
        """
        if not self._redis.is_connected:
            return None

        key = self._make_key(topic_id)
        cached = await self._redis.get(key)

        if cached:
            try:
                return json.loads(cached)
            except json.JSONDecodeError:
                return None

        return None

    async def set(self, topic_id: str, data: dict[str, Any]) -> bool:
        """Cache topic data.

        Args:
            topic_id: Topic ID
            data: Topic data to cache

        Returns:
            True if cached successfully
        """
        if not self._redis.is_connected:
            return False

        key = self._make_key(topic_id)
        return await self._redis.set(key, json.dumps(data), ex=self._ttl)

    async def invalidate(self, topic_id: str) -> bool:
        """Invalidate cached topic.

        Args:
            topic_id: Topic ID to invalidate

        Returns:
            True if invalidated successfully
        """
        if not self._redis.is_connected:
            return False

        key = self._make_key(topic_id)
        return await self._redis.delete(key)
