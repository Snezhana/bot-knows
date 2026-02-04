"""Claude import adapter for bot_knows.

This module provides an adapter for parsing Claude export files.
"""

from typing import Any, override

from bot_knows.importers.base import ChatImportAdapter
from bot_knows.importers.registry import ImportAdapterRegistry
from bot_knows.models.ingest import ChatIngest, IngestMessage


@ImportAdapterRegistry.register
class ClaudeAdapter(ChatImportAdapter):
    """Adapter for Claude export format.

    Parses the JSON export from Claude's export feature.
    The exact format may vary based on Claude version.

    Expected format:
        {
            "conversations": [
                {
                    "uuid": "conversation-id",
                    "name": "Chat Title",
                    "created_at": "2024-01-01T00:00:00Z",
                    "chat_messages": [
                        {
                            "uuid": "message-id",
                            "sender": "human",
                            "text": "Hello",
                            "created_at": "2024-01-01T00:00:00Z"
                        }
                    ]
                }
            ]
        }
    """

    @property
    @override
    def source_name(self) -> str:
        return "claude"

    @override
    def parse(self, raw_export: dict[str, Any]) -> list[ChatIngest]:
        """Parse Claude export format.

        Args:
            raw_export: Raw JSON data

        Returns:
            List of ChatIngest objects
        """
        chats: list[ChatIngest] = []

        # Handle different possible formats
        if isinstance(raw_export, list):
            conversations = raw_export
        else:
            conversations = raw_export.get("conversations", [])

        for conv in conversations:
            chat_ingest = self._parse_conversation(conv)
            if chat_ingest and chat_ingest.has_messages:
                chats.append(chat_ingest)

        return chats

    def _parse_conversation(self, conv: dict[str, Any]) -> ChatIngest | None:
        """Parse a single conversation."""
        conv_id = conv.get("uuid", conv.get("id", ""))
        title = conv.get("name", conv.get("title"))

        # Parse timestamp
        created_at = conv.get("created_at", conv.get("create_time", 0))
        create_time = self._parse_timestamp(created_at)

        # Get messages
        raw_messages = conv.get("chat_messages", conv.get("messages", []))
        messages: list[IngestMessage] = []

        for raw_msg in raw_messages:
            msg = self._parse_message(raw_msg, conv_id, create_time)
            if msg:
                messages.append(msg)

        # Sort messages by timestamp
        messages.sort(key=lambda m: m.timestamp)

        return ChatIngest(
            source="claude",
            imported_chat_timestamp=create_time,
            title=title,
            messages=messages,
            provider="claude",
            conversation_id=conv_id,
        )

    def _parse_message(
        self,
        raw_msg: dict[str, Any],
        conv_id: str,
        default_timestamp: int,
    ) -> IngestMessage | None:
        """Parse a single message."""
        # Get role (Claude uses 'sender' with 'human'/'assistant')
        sender = raw_msg.get("sender", raw_msg.get("role", ""))
        role = self._normalize_role(sender)
        if not role:
            return None

        # Get content
        content = raw_msg.get("text", raw_msg.get("content", ""))
        if isinstance(content, list):
            # Handle content blocks format
            content = self._extract_text_from_blocks(content)
        if not content:
            return None

        # Get timestamp
        created_at = raw_msg.get("created_at", raw_msg.get("timestamp"))
        timestamp = self._parse_timestamp(created_at)
        if timestamp == 0:
            timestamp = default_timestamp

        return IngestMessage(
            role=role,  # type: ignore[arg-type]
            content=content,
            timestamp=timestamp,
            chat_id=conv_id,
        )

    def _normalize_role(self, sender: str) -> str | None:
        """Normalize sender/role to standard roles."""
        sender_lower = sender.lower()
        if sender_lower in ("human", "user"):
            return "user"
        if sender_lower in ("assistant", "ai", "claude"):
            return "assistant"
        if sender_lower == "system":
            return "system"
        return None

    def _extract_text_from_blocks(self, blocks: list[Any]) -> str:
        """Extract text from content blocks format."""
        texts = []
        for block in blocks:
            if isinstance(block, str):
                texts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                texts.append(block.get("text", ""))
        return "\n".join(texts).strip()

    def _parse_timestamp(self, value: Any) -> int:
        """Parse timestamp to epoch seconds."""
        if value is None:
            return 0
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            # Try to parse ISO format
            try:
                from datetime import datetime

                dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
                return int(dt.timestamp())
            except (ValueError, TypeError):
                return 0
        return 0
