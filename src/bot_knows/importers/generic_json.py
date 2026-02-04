"""Generic JSON import adapter for bot_knows.

This module provides a flexible adapter for custom JSON formats.
"""

from collections.abc import Callable
from typing import Any, override

from bot_knows.importers.base import ChatImportAdapter
from bot_knows.importers.registry import ImportAdapterRegistry
from bot_knows.models.ingest import ChatIngest, IngestMessage


@ImportAdapterRegistry.register
class GenericJSONAdapter(ChatImportAdapter):
    """Flexible adapter for custom JSON formats.

    This adapter can handle various JSON structures by allowing
    customization of field mappings.

    Expected default format:
        {
            "chats": [
                {
                    "id": "chat-id",
                    "title": "Chat Title",
                    "timestamp": 1704067200,
                    "messages": [
                        {
                            "role": "user",
                            "content": "Hello",
                            "timestamp": 1704067200
                        }
                    ]
                }
            ]
        }

    Or simpler flat format:
        [
            {
                "role": "user",
                "content": "Hello",
                "timestamp": 1704067200,
                "chat_id": "chat-1"
            }
        ]
    """

    def __init__(
        self,
        chats_key: str = "chats",
        messages_key: str = "messages",
        role_key: str = "role",
        content_key: str = "content",
        timestamp_key: str = "timestamp",
        chat_id_key: str = "id",
        title_key: str = "title",
        role_mapping: dict[str, str] | None = None,
        content_extractor: Callable[[Any], str] | None = None,
    ) -> None:
        """Initialize adapter with field mappings.

        Args:
            chats_key: Key for chats array in root object
            messages_key: Key for messages array in chat object
            role_key: Key for role in message object
            content_key: Key for content in message object
            timestamp_key: Key for timestamp in message/chat object
            chat_id_key: Key for chat ID in chat object
            title_key: Key for title in chat object
            role_mapping: Custom role mapping (e.g., {"human": "user"})
            content_extractor: Custom function to extract content from message
        """
        self._chats_key = chats_key
        self._messages_key = messages_key
        self._role_key = role_key
        self._content_key = content_key
        self._timestamp_key = timestamp_key
        self._chat_id_key = chat_id_key
        self._title_key = title_key
        self._role_mapping = role_mapping or {}
        self._content_extractor = content_extractor

    @property
    @override
    def source_name(self) -> str:
        return "generic_json"

    @override
    def parse(self, raw_export: dict[str, Any]) -> list[ChatIngest]:
        """Parse generic JSON format.

        Supports:
        1. Object with 'chats' array containing chat objects with 'messages'
        2. Direct array of chat objects
        3. Flat array of messages with chat_id field

        Args:
            raw_export: Raw JSON data

        Returns:
            List of ChatIngest objects
        """
        # Try different formats
        if isinstance(raw_export, list):
            return self._parse_list(raw_export)

        if self._chats_key in raw_export:
            return self._parse_chats_object(raw_export)

        # Try to interpret as single chat
        return self._parse_single_chat(raw_export)

    def _parse_list(self, items: list[Any]) -> list[ChatIngest]:
        """Parse list format (could be chats or flat messages)."""
        if not items:
            return []

        # Check if first item looks like a chat or a message
        first = items[0]
        if isinstance(first, dict) and self._messages_key in first:
            # List of chat objects
            return [self._parse_chat(chat) for chat in items if chat]

        # Assume flat list of messages - group by chat_id
        return self._parse_flat_messages(items)

    def _parse_chats_object(self, data: dict[str, Any]) -> list[ChatIngest]:
        """Parse object with chats array."""
        chats = data.get(self._chats_key, [])
        return [self._parse_chat(chat) for chat in chats if chat]

    def _parse_single_chat(self, data: dict[str, Any]) -> list[ChatIngest]:
        """Parse as single chat object."""
        chat = self._parse_chat(data)
        return [chat] if chat.has_messages else []

    def _parse_chat(self, chat: dict[str, Any]) -> ChatIngest:
        """Parse a single chat object."""
        chat_id = str(chat.get(self._chat_id_key, ""))
        title = chat.get(self._title_key)
        timestamp = self._parse_timestamp(chat.get(self._timestamp_key, 0))

        raw_messages = chat.get(self._messages_key, [])
        messages = [
            msg
            for msg in (self._parse_message(m, chat_id, timestamp) for m in raw_messages)
            if msg is not None
        ]

        messages.sort(key=lambda m: m.timestamp)

        return ChatIngest(
            source="generic_json",
            imported_chat_timestamp=timestamp,
            title=title,
            messages=messages,
            conversation_id=chat_id,
        )

    def _parse_flat_messages(self, messages: list[Any]) -> list[ChatIngest]:
        """Parse flat list of messages into chats grouped by chat_id."""
        from collections import defaultdict

        # Group messages by chat_id
        grouped: dict[str, list[IngestMessage]] = defaultdict(list)

        for raw_msg in messages:
            if not isinstance(raw_msg, dict):
                continue

            chat_id = str(raw_msg.get("chat_id", raw_msg.get(self._chat_id_key, "default")))
            msg = self._parse_message(raw_msg, chat_id, 0)
            if msg:
                grouped[chat_id].append(msg)

        # Create ChatIngest for each group
        chats: list[ChatIngest] = []
        for chat_id, msgs in grouped.items():
            msgs.sort(key=lambda m: m.timestamp)
            min_timestamp = msgs[0].timestamp if msgs else 0

            chats.append(
                ChatIngest(
                    source="generic_json",
                    imported_chat_timestamp=min_timestamp,
                    title=None,
                    messages=msgs,
                    conversation_id=chat_id,
                )
            )

        return chats

    def _parse_message(
        self,
        raw_msg: dict[str, Any],
        chat_id: str,
        default_timestamp: int,
    ) -> IngestMessage | None:
        """Parse a single message."""
        # Get role
        raw_role = raw_msg.get(self._role_key, "")
        role = self._normalize_role(raw_role)
        if not role:
            return None

        # Get content
        if self._content_extractor:
            content = self._content_extractor(raw_msg)
        else:
            content = raw_msg.get(self._content_key, "")
            if isinstance(content, list):
                content = "\n".join(str(c) for c in content)

        if not content:
            return None

        # Get timestamp
        timestamp = self._parse_timestamp(raw_msg.get(self._timestamp_key))
        if timestamp == 0:
            timestamp = default_timestamp

        return IngestMessage(
            role=role,  # type: ignore[arg-type]
            content=str(content),
            timestamp=timestamp,
            chat_id=chat_id,
        )

    def _normalize_role(self, role: str) -> str | None:
        """Normalize role to standard values."""
        role_lower = str(role).lower()

        # Check custom mapping first
        if role_lower in self._role_mapping:
            return self._role_mapping[role_lower]

        # Standard mappings
        if role_lower in ("user", "human"):
            return "user"
        if role_lower in ("assistant", "ai", "bot"):
            return "assistant"
        if role_lower == "system":
            return "system"

        return None

    def _parse_timestamp(self, value: Any) -> int:
        """Parse timestamp to epoch seconds."""
        if value is None:
            return 0
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            try:
                # Try numeric string
                return int(float(value))
            except ValueError:
                pass
            try:
                # Try ISO format
                from datetime import datetime

                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return int(dt.timestamp())
            except (ValueError, TypeError):
                pass
        return 0
