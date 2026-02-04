"""Redis infrastructure for bot_knows (optional)."""

from bot_knows.infra.redis.cache import EmbeddingCache
from bot_knows.infra.redis.client import RedisClient

__all__ = ["EmbeddingCache", "RedisClient"]
