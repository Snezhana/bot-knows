"""Import adapter registry for bot_knows.

This module provides a registry for dynamically registering
and looking up chat import adapters.
"""

from bot_knows.importers.base import ChatImportAdapter

__all__ = [
    "ImportAdapterRegistry",
]


class ImportAdapterRegistry:
    """Registry for chat import adapters.

    Provides a centralized registry for import adapters, allowing
    dynamic registration and lookup by source name.

    Example:
        # Register an adapter (as decorator)
        @ImportAdapterRegistry.register
        class MyAdapter(ChatImportAdapter):
            ...

        # Or register manually
        ImportAdapterRegistry.register(MyAdapter)

        # Create an adapter instance
        adapter = ImportAdapterRegistry.create("my_source")
        chats = adapter.parse(raw_data)
    """

    _adapters: dict[str, type[ChatImportAdapter]] = {}  # noqa: RUF012

    @classmethod
    def register(
        cls,
        adapter_cls: type[ChatImportAdapter],
    ) -> type[ChatImportAdapter]:
        """Register an adapter class.

        Can be used as a decorator or called directly.

        Args:
            adapter_cls: Adapter class to register

        Returns:
            The adapter class (for use as decorator)

        Raises:
            ValueError: If adapter source_name is already registered
        """
        # Create instance to get source_name
        instance = adapter_cls()
        source_name = instance.source_name

        if source_name in cls._adapters:
            raise ValueError(f"Adapter already registered for source: {source_name}")

        cls._adapters[source_name] = adapter_cls
        return adapter_cls

    @classmethod
    def get(cls, source_name: str) -> type[ChatImportAdapter]:
        """Get adapter class by source name.

        Args:
            source_name: Source identifier to look up

        Returns:
            Adapter class

        Raises:
            KeyError: If no adapter registered for source
        """
        if source_name not in cls._adapters:
            available = ", ".join(cls._adapters.keys()) or "none"
            raise KeyError(
                f"No adapter registered for source: {source_name}. Available: {available}"
            )
        return cls._adapters[source_name]

    @classmethod
    def create(cls, source_name: str, **kwargs: object) -> ChatImportAdapter:
        """Create adapter instance by source name.

        Args:
            source_name: Source identifier to look up
            **kwargs: Arguments to pass to adapter constructor

        Returns:
            Adapter instance
        """
        adapter_cls = cls.get(source_name)
        return adapter_cls(**kwargs)

    @classmethod
    def list_sources(cls) -> list[str]:
        """List all registered source names.

        Returns:
            List of registered source identifiers
        """
        return list(cls._adapters.keys())

    @classmethod
    def is_registered(cls, source_name: str) -> bool:
        """Check if a source has a registered adapter.

        Args:
            source_name: Source identifier to check

        Returns:
            True if registered, False otherwise
        """
        return source_name in cls._adapters

    @classmethod
    def clear(cls) -> None:
        """Clear all registered adapters.

        Primarily for testing purposes.
        """
        cls._adapters.clear()
