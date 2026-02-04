"""Integration tests for bot_knows pipeline."""

import pytest
from pathlib import Path

from bot_knows.importers.chatgpt import ChatGPTAdapter
from bot_knows.importers.claude import ClaudeAdapter
from bot_knows.services.message_builder import MessageBuilder


# Path to fixtures
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


class TestChatGPTImportPipeline:
    """Integration tests for ChatGPT import pipeline."""

    def test_parse_fixture_file(self) -> None:
        """Test parsing actual ChatGPT export fixture."""
        adapter = ChatGPTAdapter()
        chats = adapter.parse_file(FIXTURES_DIR / "chatgpt_export.json")

        assert len(chats) == 2

        # First chat - Python async
        first_chat = chats[0]
        assert first_chat.title == "Python Async Programming Help"
        assert first_chat.source == "chatgpt"
        assert len(first_chat.messages) == 4

        # Second chat - Math question
        second_chat = chats[1]
        assert second_chat.title == "Quick Math Question"
        assert len(second_chat.messages) == 2

    def test_message_pairing(self) -> None:
        """Test that messages are correctly paired."""
        adapter = ChatGPTAdapter()
        chats = adapter.parse_file(FIXTURES_DIR / "chatgpt_export.json")

        builder = MessageBuilder()
        messages = builder.build(chats[0].messages, "test-chat")

        # 4 ingest messages should become 2 message pairs
        assert len(messages) == 2

        # First pair
        assert "async/await" in messages[0].user_content
        assert "asyncio" in messages[0].assistant_content

        # Second pair
        assert "gather" in messages[1].user_content.lower()
        assert "gather" in messages[1].assistant_content.lower()


class TestClaudeImportPipeline:
    """Integration tests for Claude import pipeline."""

    def test_parse_fixture_file(self) -> None:
        """Test parsing actual Claude export fixture."""
        adapter = ClaudeAdapter()
        chats = adapter.parse_file(FIXTURES_DIR / "claude_export.json")

        assert len(chats) == 1

        chat = chats[0]
        assert chat.title == "Database Design Discussion"
        assert chat.source == "claude"
        assert len(chat.messages) == 4

    def test_message_pairing(self) -> None:
        """Test that messages are correctly paired."""
        adapter = ClaudeAdapter()
        chats = adapter.parse_file(FIXTURES_DIR / "claude_export.json")

        builder = MessageBuilder()
        messages = builder.build(chats[0].messages, "test-chat")

        # 4 ingest messages should become 2 message pairs
        assert len(messages) == 2

        # First pair about schema
        assert "database" in messages[0].user_content.lower()
        assert "users" in messages[0].assistant_content.lower()

        # Second pair about indexes
        assert "index" in messages[1].user_content.lower()
        assert "index" in messages[1].assistant_content.lower()


class TestCrossAdapterConsistency:
    """Test that different adapters produce consistent output."""

    def test_message_structure_consistency(self) -> None:
        """Test that all adapters produce valid IngestMessage objects."""
        chatgpt_adapter = ChatGPTAdapter()
        claude_adapter = ClaudeAdapter()

        chatgpt_chats = chatgpt_adapter.parse_file(FIXTURES_DIR / "chatgpt_export.json")
        claude_chats = claude_adapter.parse_file(FIXTURES_DIR / "claude_export.json")

        for chat in chatgpt_chats + claude_chats:
            # All chats should have required fields
            assert chat.source is not None
            assert chat.imported_chat_timestamp >= 0

            for msg in chat.messages:
                # All messages should have valid roles
                assert msg.role in ("user", "assistant", "system")
                assert msg.content is not None
                assert msg.timestamp >= 0
