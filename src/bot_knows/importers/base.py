"""Base import adapter for bot_knows.

This module defines the abstract base class for chat import adapters.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, BinaryIO

from bot_knows.models.ingest import ChatIngest

__all__ = [
    "ChatImportAdapter",
]


class ChatImportAdapter(ABC):
    """Abstract base class for chat import adapters.

    Import adapters are responsible for parsing provider-specific
    export formats into the canonical ChatIngest model.

    Important: Adapters must NOT persist data or mutate any state.
    They only normalize data.

    Example:
        class MyAdapter(ChatImportAdapter):
            @property
            def source_name(self) -> str:
                return "my_source"

            def parse(self, raw_export: dict) -> list[ChatIngest]:
                # Parse and return ChatIngest objects
                ...
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Return unique identifier for this import source.

        This name is used to identify the source in ChatIngest.source
        and for adapter registry lookup.

        Returns:
            Source identifier string (e.g., "chatgpt", "claude")
        """
        ...

    @abstractmethod
    def parse(self, raw_export: dict[str, Any]) -> list[ChatIngest]:
        """Parse raw export data into ChatIngest objects.

        This method must be pure - it should not persist data,
        generate IDs, classify, or mutate any state.

        Args:
            raw_export: Raw JSON data from the export file

        Returns:
            List of ChatIngest objects (one export may contain multiple chats)

        Raises:
            ValueError: If the export format is invalid
        """
        ...

    def parse_file(self, path: Path | str) -> list[ChatIngest]:
        """Parse from file path.

        Convenience method that loads JSON from file and calls parse().

        Args:
            path: Path to the export JSON file

        Returns:
            List of ChatIngest objects
        """
        import json

        path = Path(path)
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return self.parse(data)

    def parse_stream(self, stream: BinaryIO) -> list[ChatIngest]:
        """Parse from file stream.

        Convenience method that loads JSON from stream and calls parse().

        Args:
            stream: Binary file stream containing JSON data

        Returns:
            List of ChatIngest objects
        """
        import json

        data = json.load(stream)
        return self.parse(data)

    def parse_string(self, json_string: str) -> list[ChatIngest]:
        """Parse from JSON string.

        Convenience method that parses JSON string and calls parse().

        Args:
            json_string: JSON string

        Returns:
            List of ChatIngest objects
        """
        import json

        data = json.loads(json_string)
        return self.parse(data)
