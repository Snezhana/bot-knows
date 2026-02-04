"""ChatGPT import adapter for bot_knows.

This module provides an adapter for parsing ChatGPT export files.
"""

from typing import Any, override

from bot_knows.importers.base import ChatImportAdapter
from bot_knows.importers.registry import ImportAdapterRegistry
from bot_knows.models.ingest import ChatIngest, IngestMessage


@ImportAdapterRegistry.register
class ChatGPTAdapter(ChatImportAdapter):
    """Adapter for ChatGPT export format (conversations.json).

    Parses the JSON export from ChatGPT's "Export data" feature.
    The export contains a list of conversations, each with a mapping
    of message nodes.

    Export format example:
        [
            {
                "id": "conversation-id",
                "title": "Chat Title",
                "create_time": 1704067200,
                "mapping": {
                    "node-id": {
                        "message": {
                            "author": {"role": "user"},
                            "content": {"parts": ["Hello"]},
                            "create_time": 1704067200
                        }
                    }
                }
            }
        ]
    """

    @property
    @override
    def source_name(self) -> str:
        return "chatgpt"

    @override
    def parse(self, raw_export: dict[str, Any]) -> list[ChatIngest]:
        """Parse ChatGPT export format.

        Args:
            raw_export: Raw JSON data (list of conversations or dict with 'conversations' key)

        Returns:
            List of ChatIngest objects
        """
        chats: list[ChatIngest] = []

        # Handle both formats: direct list or dict with 'conversations' key
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
        conv_id = conv.get("id", "")
        title = conv.get("title")
        create_time = self._parse_timestamp(conv.get("create_time", 0))

        messages: list[IngestMessage] = []
        mapping = conv.get("mapping", {})

        for node in mapping.values():
            msg = self._parse_message_node(node, conv_id, create_time)
            if msg:
                messages.append(msg)

        # Sort messages by timestamp
        messages.sort(key=lambda m: m.timestamp)

        return ChatIngest(
            source="chatgpt",
            imported_chat_timestamp=create_time,
            title=title,
            messages=messages,
            provider="chatgpt",
            conversation_id=conv_id,
        )

    def _parse_message_node(
        self,
        node: dict[str, Any],
        conv_id: str,
        default_timestamp: int,
    ) -> IngestMessage | None:
        """Parse a message node from the mapping."""
        msg = node.get("message")
        if not msg:
            return None

        # Get role
        author = msg.get("author", {})
        role = author.get("role", "")
        if role not in ("user", "assistant", "system"):
            return None

        # Get content
        content = self._extract_content(msg)
        if not content:
            return None

        # Get timestamp
        timestamp = self._parse_timestamp(msg.get("create_time"))
        if timestamp == 0:
            timestamp = default_timestamp

        return IngestMessage(
            role=role,  # type: ignore[arg-type]
            content=content,
            timestamp=timestamp,
            chat_id=conv_id,
        )

    def _extract_content(self, msg: dict[str, Any]) -> str:
        """Extract text content from message."""
        content_obj = msg.get("content", {})

        # Handle different content formats
        if isinstance(content_obj, str):
            return content_obj

        if isinstance(content_obj, dict):
            parts = content_obj.get("parts", [])
            # Filter and join text parts
            text_parts = [
                str(p) for p in parts
                if p and isinstance(p, (str, int, float))
            ]
            return " ".join(text_parts).strip()

        return ""

    def _parse_timestamp(self, value: Any) -> int:
        """Parse timestamp to epoch seconds."""
        if value is None:
            return 0
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        return 0
