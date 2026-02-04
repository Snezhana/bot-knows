"""Configuration management for bot_knows.

This module provides typed configuration classes using pydantic-settings.
Configuration is loaded from environment variables with optional .env file support.
"""

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = [
    "MongoSettings",
    "Neo4jSettings",
    "RedisSettings",
    "LLMSettings",
    "BotKnowsConfig",
]


class MongoSettings(BaseSettings):
    """MongoDB connection settings."""

    model_config = SettingsConfigDict(
        env_prefix="BOT_KNOWS_MONGO_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    uri: SecretStr = SecretStr("mongodb://localhost:27017")
    database: str = "bot_knows"
    collection_prefix: str = ""


class Neo4jSettings(BaseSettings):
    """Neo4j connection settings."""

    model_config = SettingsConfigDict(
        env_prefix="BOT_KNOWS_NEO4J_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    uri: str = "bolt://localhost:7687"
    username: str = "neo4j"
    password: SecretStr = SecretStr("password")
    database: str = "neo4j"


class RedisSettings(BaseSettings):
    """Redis connection settings (optional).

    If url is not configured or connection fails, caching will be disabled.
    """

    model_config = SettingsConfigDict(
        env_prefix="BOT_KNOWS_REDIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    url: str | None = None
    enabled: bool = True  # Can be explicitly disabled


class LLMSettings(BaseSettings):
    """LLM provider settings."""

    model_config = SettingsConfigDict(
        env_prefix="BOT_KNOWS_LLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    provider: str = "openai"  # "openai" or "anthropic"
    api_key: SecretStr | None = None
    model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536


class BotKnowsConfig(BaseSettings):
    """Main configuration aggregating all settings.

    Example usage:
        config = BotKnowsConfig()
        mongo_uri = config.mongo.uri.get_secret_value()
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Component settings (nested)
    mongo: MongoSettings = MongoSettings()
    neo4j: Neo4jSettings = Neo4jSettings()
    redis: RedisSettings = RedisSettings()
    llm: LLMSettings = LLMSettings()

    # Deduplication thresholds
    dedup_high_threshold: float = 0.92
    dedup_low_threshold: float = 0.80

    # Recall settings
    recall_stability_k: float = 0.1
    recall_semantic_boost: float = 0.1
    decay_batch_interval_hours: int = 24

    @property
    def redis_enabled(self) -> bool:
        """Check if Redis caching is enabled and configured."""
        return self.redis.enabled and self.redis.url is not None
