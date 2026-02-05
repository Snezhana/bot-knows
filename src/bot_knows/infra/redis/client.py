"""Redis client for bot_knows.

This module provides an optional async Redis client wrapper.
Redis is used for caching and is optional - if not configured,
caching will be disabled gracefully.
"""

from typing import TYPE_CHECKING, Any

from bot_knows.config import RedisSettings
from bot_knows.logging import get_logger
from bot_knows.utils.lazy_import import lazy_import

if TYPE_CHECKING:
    from redis.asyncio import Redis

__all__ = [
    "RedisClient",
]

logger = get_logger(__name__)

get_async_redis = lazy_import("redis.asyncio", "Redis")


class RedisClient:
    """Async Redis client wrapper (optional).

    Provides connection management for Redis caching.
    If Redis is not configured or unavailable, operations
    will fail gracefully.

    Example:
        client = RedisClient(settings)
        if await client.connect():
            await client.set("key", "value")
            value = await client.get("key")
        await client.disconnect()
    """

    def __init__(self, settings: RedisSettings) -> None:
        """Initialize client with settings.

        Args:
            settings: Redis connection settings
        """
        self._settings = settings
        self._redis = None  # type: ignore[type-arg]
        self._connected = False

    @property
    def is_enabled(self) -> bool:
        """Check if Redis is enabled in configuration."""
        return self._settings.enabled and self._settings.url is not None

    @property
    def is_connected(self) -> bool:
        """Check if connected to Redis."""
        return self._connected

    async def connect(self) -> bool:
        """Initialize connection to Redis.

        Returns:
            True if connected successfully, False otherwise
        """
        if self._redis is not None:
            return self._connected

        if not self.is_enabled:
            logger.info("redis_disabled", reason="not configured")
            return False

        try:
            Redis = get_async_redis()  # noqa: N806
            self._redis = Redis.from_url(
                self._settings.url,  # type: ignore[arg-type]
                decode_responses=True,
            )
            # Verify connection
            await self._redis.ping()
            self._connected = True
            logger.info("connected_to_redis", url=self._settings.url)
            return True
        except Exception as e:
            logger.warning(
                "redis_connection_failed",
                error=str(e),
                reason="Redis unavailable, caching disabled",
            )
            self._redis = None
            self._connected = False
            return False

    async def disconnect(self) -> None:
        """Close connection to Redis."""
        if self._redis:
            await self._redis.close()
            self._redis = None
            self._connected = False
            logger.info("disconnected_from_redis")

    @property
    def client(self) -> "Redis | None":  # type: ignore[type-arg]
        """Get Redis client instance.

        Returns:
            Redis client if connected, None otherwise
        """
        return self._redis if self._connected else None

    async def get(self, key: str) -> str | None:
        """Get value from Redis.

        Args:
            key: Cache key

        Returns:
            Cached value or None if not found/not connected
        """
        if not self._connected or not self._redis:
            return None
        try:
            result = await self._redis.get(key)
            return result  # type: ignore[return-value]
        except Exception as e:
            logger.debug("redis_get_error", key=key, error=str(e))
            return None

    async def set(
        self,
        key: str,
        value: str,
        ex: int | None = None,
    ) -> bool:
        """Set value in Redis.

        Args:
            key: Cache key
            value: Value to cache
            ex: Expiration time in seconds

        Returns:
            True if successful, False otherwise
        """
        if not self._connected or not self._redis:
            return False
        try:
            await self._redis.set(key, value, ex=ex)
            return True
        except Exception as e:
            logger.debug("redis_set_error", key=key, error=str(e))
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from Redis.

        Args:
            key: Cache key to delete

        Returns:
            True if successful, False otherwise
        """
        if not self._connected or not self._redis:
            return False
        try:
            await self._redis.delete(key)
            return True
        except Exception as e:
            logger.debug("redis_delete_error", key=key, error=str(e))
            return False

    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis.

        Args:
            key: Cache key to check

        Returns:
            True if exists, False otherwise
        """
        if not self._connected or not self._redis:
            return False
        try:
            result = await self._redis.exists(key)
            return bool(result)
        except Exception as e:
            logger.debug("redis_exists_error", key=key, error=str(e))
            return False

    async def __aenter__(self) -> "RedisClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()
