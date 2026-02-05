from collections.abc import Callable
from importlib import import_module


def lazy_import(
    module_name: str,
    name: str | None = None,
) -> Callable[[], object]:
    """Lazily import a module or an attribute from a module."""

    def _load() -> object:
        mod = import_module(module_name)
        return getattr(mod, name) if name else mod

    return _load
