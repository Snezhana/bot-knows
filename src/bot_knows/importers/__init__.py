"""Import adapters for bot_knows.

This module exports the import adapter base class and registry.
"""

from bot_knows.importers.base import ChatImportAdapter
from bot_knows.importers.registry import ImportAdapterRegistry

__all__ = [
    "ChatImportAdapter",
    "ImportAdapterRegistry",
]
