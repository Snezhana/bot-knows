"""Hashing utilities for bot_knows.

This module provides deterministic hash functions for generating
stable identifiers for chats, messages, topics, and evidence.
"""

import hashlib
from typing import Any

__all__ = [
    "generate_chat_id",
    "generate_evidence_id",
    "generate_message_id",
    "generate_topic_id",
    "hash_text",
]


def hash_text(text: str) -> str:
    """Generate SHA256 hash of text.

    Args:
        text: Input text to hash

    Returns:
        Hexadecimal SHA256 hash string
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def generate_chat_id(title: str, source: str, timestamp: int) -> str:
    """Generate deterministic chat ID.

    The chat ID is a SHA256 hash of: title + source + timestamp.
    This ensures the same chat imported multiple times gets the same ID.

    Args:
        title: Chat title
        source: Import source (e.g., "chatgpt", "claude")
        timestamp: Chat creation timestamp (epoch seconds)

    Returns:
        Hexadecimal SHA256 hash string
    """
    combined = f"{title}|{source}|{timestamp}"
    return hash_text(combined)


def generate_message_id(
    chat_id: str,
    user_content: str,
    assistant_content: str,
    timestamp: int,
) -> str:
    """Generate deterministic message ID.

    The message ID is based on chat_id + content + timestamp to ensure
    uniqueness within a chat while being deterministic.

    Args:
        chat_id: Parent chat ID
        user_content: User message content
        assistant_content: Assistant response content
        timestamp: Message timestamp (epoch seconds)

    Returns:
        Hexadecimal SHA256 hash string
    """
    combined = f"{chat_id}|{user_content}|{assistant_content}|{timestamp}"
    return hash_text(combined)


def generate_topic_id(canonical_name: str, source_message_id: str) -> str:
    """Generate deterministic topic ID.

    The topic ID combines the canonical name with the first message
    that introduced it, ensuring uniqueness.

    Args:
        canonical_name: Canonical topic name
        source_message_id: ID of the message that first introduced this topic

    Returns:
        Hexadecimal SHA256 hash string
    """
    combined = f"topic|{canonical_name}|{source_message_id}"
    return hash_text(combined)


def generate_evidence_id(
    topic_id: str,
    extracted_name: str,
    source_message_id: str,
    timestamp: int,
) -> str:
    """Generate deterministic evidence ID.

    Evidence IDs are unique per extraction event.

    Args:
        topic_id: Parent topic ID
        extracted_name: Raw extracted topic name
        source_message_id: Source message ID
        timestamp: Extraction timestamp (epoch seconds)

    Returns:
        Hexadecimal SHA256 hash string
    """
    combined = f"evidence|{topic_id}|{extracted_name}|{source_message_id}|{timestamp}"
    return hash_text(combined)


def stable_hash(*args: Any) -> str:
    """Generate a stable hash from multiple arguments.

    Converts all arguments to strings and joins them with pipe separator.
    Useful for creating composite keys.

    Args:
        *args: Values to include in the hash

    Returns:
        Hexadecimal SHA256 hash string
    """
    combined = "|".join(str(arg) for arg in args)
    return hash_text(combined)
