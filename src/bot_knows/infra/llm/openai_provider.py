"""OpenAI LLM provider for bot_knows.

This module provides the OpenAI implementation of LLM and embedding interfaces.
"""

import json
from typing import Any, Self

import numpy as np
from openai import AsyncOpenAI

from bot_knows.config import LLMSettings
from bot_knows.interfaces.embedding import EmbeddingServiceInterface
from bot_knows.interfaces.llm import LLMInterface
from bot_knows.logging import get_logger
from bot_knows.models.chat import ChatCategory

__all__ = [
    "OpenAIProvider",
]

logger = get_logger(__name__)


class OpenAIProvider(LLMInterface, EmbeddingServiceInterface):
    """OpenAI implementation of LLM and embedding interfaces.

    Provides chat classification, topic extraction, and embedding
    generation using OpenAI's API.
    """

    config_class = LLMSettings

    def __init__(self, settings: LLMSettings) -> None:
        """Initialize OpenAI provider.

        Args:
            settings: LLM configuration settings
        """
        self._settings = settings
        api_key = settings.api_key.get_secret_value() if settings.api_key else None
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = settings.model
        self._embedding_model = settings.embedding_model
        self._embedding_dimensions = settings.embedding_dimensions

    @classmethod
    async def from_config(cls, config: LLMSettings) -> Self:
        """Factory method for BotKnows instantiation.

        Args:
            config: LLM settings

        Returns:
            OpenAIProvider instance
        """
        return cls(config)

    @classmethod
    async def from_dict(cls, config: dict[str, Any]) -> Self:
        """Factory method for custom config dict.

        Args:
            config: Dictionary with LLM settings

        Returns:
            OpenAIProvider instance
        """
        settings = LLMSettings(**config)
        return cls(settings)

    async def close(self) -> None:
        """Close resources (no-op for OpenAI)."""
        pass

    # Embedding interface
    async def embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        response = await self._client.embeddings.create(
            model=self._embedding_model,
            input=text,
        )
        return response.data[0].embedding

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
        response = await self._client.embeddings.create(
            model=self._embedding_model,
            input=texts,
        )
        sorted_data = sorted(response.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    async def similarity(self, embedding1: list[float], embedding2: list[float]) -> float:
        """Compute cosine similarity between embeddings."""
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    # LLM interface
    async def classify_chat(
        self,
        first_pair: tuple[str, str],
        last_pair: tuple[str, str],
    ) -> tuple[ChatCategory, list[str]]:
        """Classify chat and extract tags."""
        system_prompt = """You are a chat classifier.
        Analyze the conversation samples and classify the chat and assign tags.

Categories: coding, research, writing, brainstorming, debugging, learning, general, other
Tags: no strick - should be subcategroy of the category.
Respond with JSON only:
{"category": "category_name", "tags": ["tag1", "tag2"]}"""

        user_content = f"""First exchange:
User: {first_pair[0][:500]}
Assistant: {first_pair[1][:500]}

Last exchange:
User: {last_pair[0][:500]}
Assistant: {last_pair[1][:500]}"""

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            content = response.choices[0].message.content or "{}"
            result = json.loads(content)
            category_str = result.get("category", "general").lower()
            try:
                category = ChatCategory(category_str)
            except ValueError:
                category = ChatCategory.GENERAL
            tags = result.get("tags", [])
            return category, tags[:5] if isinstance(tags, list) else []
        except Exception as e:
            logger.warning("chat_classification_failed", error=str(e))
            return ChatCategory.GENERAL, []

    async def extract_topics(
        self,
        user_content: str,
        assistant_content: str,
    ) -> list[tuple[str, float]]:
        """Extract topic candidates from message pair."""
        system_prompt = """Extract key topics from this conversation. Use concise canonical names.

Respond with JSON only:
{"topics": [{"name": "topic_name", "confidence": 0.9}]}

Extract 0-5 topics."""

        user_prompt = f"User: {user_content[:1000]}\n\nAssistant: {assistant_content[:1000]}"

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                temperature=0.3,
            )
            content = response.choices[0].message.content or "{}"
            result = json.loads(content)
            topics = result.get("topics", [])
            return [
                (t["name"], float(t.get("confidence", 0.5)))
                for t in topics
                if isinstance(t, dict) and "name" in t
            ][:5]
        except Exception as e:
            logger.warning("topic_extraction_failed", error=str(e))
            return []

    async def normalize_topic_name(self, extracted_name: str) -> str:
        """Normalize topic name to canonical form."""
        # Simple normalization: lowercase, strip, limit length
        normalized = extracted_name.strip().lower()
        # Capitalize first letter of each word
        normalized = " ".join(word.capitalize() for word in normalized.split())
        return normalized[:100]
