"""Unit tests for bot_knows importers."""

import pytest

from bot_knows.importers.chatgpt import ChatGPTAdapter
from bot_knows.importers.claude import ClaudeAdapter
from bot_knows.importers.generic_json import GenericJSONAdapter
from bot_knows.importers.registry import ImportAdapterRegistry


class TestChatGPTAdapter:
    """Tests for ChatGPT import adapter."""

    def test_source_name(self) -> None:
        adapter = ChatGPTAdapter()
        assert adapter.source_name == "chatgpt"

    def test_parse_single_conversation(self) -> None:
        export = [
            {
                "id": "conv1",
                "title": "Test Chat",
                "create_time": 1704067200,
                "mapping": {
                    "node1": {
                        "message": {
                            "author": {"role": "user"},
                            "content": {"parts": ["Hello"]},
                            "create_time": 1704067200,
                        }
                    },
                    "node2": {
                        "message": {
                            "author": {"role": "assistant"},
                            "content": {"parts": ["Hi there!"]},
                            "create_time": 1704067201,
                        }
                    },
                },
            }
        ]

        adapter = ChatGPTAdapter()
        chats = adapter.parse(export)

        assert len(chats) == 1
        assert chats[0].title == "Test Chat"
        assert chats[0].source == "chatgpt"
        assert len(chats[0].messages) == 2
        assert chats[0].messages[0].role == "user"
        assert chats[0].messages[0].content == "Hello"

    def test_parse_empty_export(self) -> None:
        adapter = ChatGPTAdapter()
        chats = adapter.parse([])
        assert len(chats) == 0

    def test_parse_filters_empty_content(self) -> None:
        export = [
            {
                "id": "conv1",
                "title": "Test",
                "create_time": 1704067200,
                "mapping": {
                    "node1": {
                        "message": {
                            "author": {"role": "user"},
                            "content": {"parts": [""]},  # Empty content
                        }
                    }
                },
            }
        ]

        adapter = ChatGPTAdapter()
        chats = adapter.parse(export)

        # Chat with no valid messages should not be included
        assert len(chats) == 0


class TestClaudeAdapter:
    """Tests for Claude import adapter."""

    def test_source_name(self) -> None:
        adapter = ClaudeAdapter()
        assert adapter.source_name == "claude"

    def test_parse_conversation(self) -> None:
        export = {
            "conversations": [
                {
                    "uuid": "conv1",
                    "name": "Test Chat",
                    "created_at": "2024-01-01T00:00:00Z",
                    "chat_messages": [
                        {
                            "sender": "human",
                            "text": "Hello",
                            "created_at": "2024-01-01T00:00:00Z",
                        },
                        {
                            "sender": "assistant",
                            "text": "Hi there!",
                            "created_at": "2024-01-01T00:00:01Z",
                        },
                    ],
                }
            ]
        }

        adapter = ClaudeAdapter()
        chats = adapter.parse(export)

        assert len(chats) == 1
        assert chats[0].source == "claude"
        assert len(chats[0].messages) == 2
        assert chats[0].messages[0].role == "user"


class TestGenericJSONAdapter:
    """Tests for generic JSON import adapter."""

    def test_source_name(self) -> None:
        adapter = GenericJSONAdapter()
        assert adapter.source_name == "generic_json"

    def test_parse_chats_format(self) -> None:
        export = {
            "chats": [
                {
                    "id": "chat1",
                    "title": "Test Chat",
                    "timestamp": 1704067200,
                    "messages": [
                        {"role": "user", "content": "Hello", "timestamp": 1704067200},
                        {"role": "assistant", "content": "Hi!", "timestamp": 1704067201},
                    ],
                }
            ]
        }

        adapter = GenericJSONAdapter()
        chats = adapter.parse(export)

        assert len(chats) == 1
        assert chats[0].title == "Test Chat"
        assert len(chats[0].messages) == 2

    def test_parse_flat_messages(self) -> None:
        export = [
            {"role": "user", "content": "Hello", "timestamp": 1704067200, "chat_id": "c1"},
            {"role": "assistant", "content": "Hi!", "timestamp": 1704067201, "chat_id": "c1"},
            {"role": "user", "content": "Bye", "timestamp": 1704067300, "chat_id": "c2"},
        ]

        adapter = GenericJSONAdapter()
        chats = adapter.parse(export)

        assert len(chats) == 2  # Two chat_ids

    def test_custom_field_mapping(self) -> None:
        export = {
            "conversations": [
                {
                    "conversation_id": "c1",
                    "name": "My Chat",
                    "msgs": [
                        {"author": "human", "text": "Hi", "ts": 1704067200},
                    ],
                }
            ]
        }

        adapter = GenericJSONAdapter(
            chats_key="conversations",
            messages_key="msgs",
            role_key="author",
            content_key="text",
            timestamp_key="ts",
            chat_id_key="conversation_id",
            title_key="name",
            role_mapping={"human": "user"},
        )
        chats = adapter.parse(export)

        assert len(chats) == 1
        assert chats[0].messages[0].role == "user"


class TestImportAdapterRegistry:
    """Tests for import adapter registry."""

    def test_builtin_adapters_registered(self) -> None:
        # Built-in adapters are registered on import
        sources = ImportAdapterRegistry.list_sources()
        assert "chatgpt" in sources
        assert "claude" in sources
        assert "generic_json" in sources

    def test_get_adapter(self) -> None:
        adapter_cls = ImportAdapterRegistry.get("chatgpt")
        assert adapter_cls == ChatGPTAdapter

    def test_get_unknown_adapter(self) -> None:
        with pytest.raises(KeyError):
            ImportAdapterRegistry.get("unknown_source")

    def test_create_adapter(self) -> None:
        adapter = ImportAdapterRegistry.create("chatgpt")
        assert isinstance(adapter, ChatGPTAdapter)

    def test_is_registered(self) -> None:
        assert ImportAdapterRegistry.is_registered("chatgpt") is True
        assert ImportAdapterRegistry.is_registered("unknown") is False
