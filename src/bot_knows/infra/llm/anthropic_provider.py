"""Anthropic LLM provider for bot_knows.

This module provides the Anthropic implementation of LLM interface.
Note: Anthropic does not provide embeddings, so this provider
requires a separate embedding service.
"""

import json
from typing import Any, Self

from anthropic import AsyncAnthropic

from bot_knows.config import LLMSettings
from bot_knows.interfaces.llm import LLMInterface
from bot_knows.logging import get_logger
from bot_knows.models.chat import ChatCategory

__all__ = [
    "AnthropicProvider",
]

logger = get_logger(__name__)


class AnthropicProvider(LLMInterface):
    """Anthropic implementation of LLM interface.

    Provides chat classification and topic extraction using
    Anthropic's Claude API.

    Note: This provider does NOT implement embedding generation.
    Use OpenAI or another embedding provider for embeddings.
    """

    config_class = LLMSettings

    def __init__(self, settings: LLMSettings) -> None:
        """Initialize Anthropic provider.

        Args:
            settings: LLM configuration settings
        """
        self._settings = settings
        api_key = settings.api_key.get_secret_value() if settings.api_key else None
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = settings.model or "claude-sonnet-4-20250514"

    @classmethod
    async def from_config(cls, config: LLMSettings) -> Self:
        """Factory method for BotKnows instantiation.

        Args:
            config: LLM settings

        Returns:
            AnthropicProvider instance
        """
        return cls(config)

    @classmethod
    async def from_dict(cls, config: dict[str, Any]) -> Self:
        """Factory method for custom config dict.

        Args:
            config: Dictionary with LLM settings

        Returns:
            AnthropicProvider instance
        """
        settings = LLMSettings(**config)
        return cls(settings)

    async def close(self) -> None:
        """Close resources (no-op for Anthropic)."""
        pass

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
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=256,
                system=system_prompt,
                messages=[{"role": "user", "content": user_content}],
            )
            content = response.content[0].text if response.content else "{}"
            # Extract JSON from response
            result = self._parse_json_response(content)
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
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=256,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
            content = response.content[0].text if response.content else "{}"
            result = self._parse_json_response(content)
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
        normalized = extracted_name.strip().lower()
        normalized = " ".join(word.capitalize() for word in normalized.split())
        return normalized[:100]

    @staticmethod
    def _parse_json_response(content: str) -> dict:
        """Parse JSON from response, handling markdown code blocks."""
        content = content.strip()
        # Remove markdown code blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last lines (```json and ```)
            content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {}
