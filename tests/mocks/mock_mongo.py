"""Mock MongoDB client for testing."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock


class MockMongoCollection:
    """Mock MongoDB collection."""

    def __init__(self) -> None:
        self._documents: dict[str, dict[str, Any]] = {}

    async def insert_one(self, document: dict[str, Any]) -> MagicMock:
        doc_id = document.get("id") or document.get("_id") or str(len(self._documents))
        self._documents[doc_id] = document
        result = MagicMock()
        result.inserted_id = doc_id
        return result

    async def replace_one(
        self,
        filter_: dict[str, Any],
        replacement: dict[str, Any],
        upsert: bool = False,
    ) -> MagicMock:
        key = list(filter_.values())[0]
        if key in self._documents or upsert:
            self._documents[key] = replacement
        result = MagicMock()
        result.modified_count = 1
        return result

    async def find_one(self, filter_: dict[str, Any]) -> dict[str, Any] | None:
        key = list(filter_.values())[0]
        return self._documents.get(key)

    async def count_documents(
        self,
        filter_: dict[str, Any],
        limit: int = 0,
    ) -> int:
        key = list(filter_.values())[0]
        return 1 if key in self._documents else 0

    def find(self, filter_: dict[str, Any] | None = None) -> "MockCursor":
        if filter_ is None:
            docs = list(self._documents.values())
        else:
            docs = [
                d for d in self._documents.values()
                if all(d.get(k) == v for k, v in filter_.items())
            ]
        return MockCursor(docs)


class MockCursor:
    """Mock MongoDB cursor."""

    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self._documents = documents

    def sort(self, *args: Any, **kwargs: Any) -> "MockCursor":
        return self

    def limit(self, n: int) -> "MockCursor":
        self._documents = self._documents[:n]
        return self

    def __aiter__(self) -> "MockCursor":
        self._index = 0
        return self

    async def __anext__(self) -> dict[str, Any]:
        if self._index >= len(self._documents):
            raise StopAsyncIteration
        doc = self._documents[self._index]
        self._index += 1
        return doc


class MockMongoClient:
    """Mock MongoDB client for testing."""

    def __init__(self) -> None:
        self._collections: dict[str, MockMongoCollection] = {}

    def __getitem__(self, name: str) -> MockMongoCollection:
        if name not in self._collections:
            self._collections[name] = MockMongoCollection()
        return self._collections[name]

    @property
    def chats(self) -> MockMongoCollection:
        return self["chats"]

    @property
    def messages(self) -> MockMongoCollection:
        return self["messages"]

    @property
    def topics(self) -> MockMongoCollection:
        return self["topics"]

    @property
    def evidence(self) -> MockMongoCollection:
        return self["topic_evidence"]

    @property
    def recall_states(self) -> MockMongoCollection:
        return self["recall_states"]

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def create_indexes(self) -> None:
        pass
