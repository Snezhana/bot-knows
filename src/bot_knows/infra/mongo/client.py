"""MongoDB client for bot_knows.

This module provides an async MongoDB client wrapper using Motor.
"""

from typing import TYPE_CHECKING, Any

from bot_knows.config import MongoSettings
from bot_knows.logging import get_logger
from bot_knows.utils.lazy_import import lazy_import

if TYPE_CHECKING:
    from motor.motor_asyncio import AsyncIOMotorCollection, AsyncIOMotorDatabase

__all__ = [
    "MongoClient",
]

logger = get_logger(__name__)

get_async_motor = lazy_import("motor.motor_asyncio", "AsyncIOMotorClient")


class MongoClient:
    """Async MongoDB client wrapper.

    Provides a connection manager and collection accessors
    for the bot_knows MongoDB database.

    Example:
        client = MongoClient(settings)
        await client.connect()

        # Access collections
        await client.chats.insert_one(chat_data)

        await client.disconnect()
    """

    def __init__(self, settings: MongoSettings) -> None:
        """Initialize client with settings.

        Args:
            settings: MongoDB connection settings
        """
        self._settings = settings
        self._client = None
        self._db = None

    async def connect(self) -> None:
        """Initialize connection to MongoDB."""
        if self._client is not None:
            return
        AsyncIOMotorClient = get_async_motor()  # noqa: N806

        uri = self._settings.uri.get_secret_value()
        self._client = AsyncIOMotorClient(uri)
        self._db = self._client[self._settings.database]

        # Verify connection
        await self._client.admin.command("ping")
        logger.info(
            "connected_to_mongodb",
            database=self._settings.database,
        )

    async def disconnect(self) -> None:
        """Close connection to MongoDB."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            logger.info("disconnected_from_mongodb")

    @property
    def db(self) -> "AsyncIOMotorDatabase[dict[str, Any]]":
        """Get database instance.

        Raises:
            RuntimeError: If not connected
        """
        if self._db is None:
            raise RuntimeError("MongoClient not connected. Call connect() first.")
        return self._db

    def _collection(self, name: str) -> "AsyncIOMotorCollection[dict[str, Any]]":
        """Get collection with optional prefix."""
        full_name = f"{self._settings.collection_prefix}{name}"
        return self.db[full_name]

    @property
    def chats(self) -> "AsyncIOMotorCollection[dict[str, Any]]":
        """Get chats collection."""
        return self._collection("chats")

    @property
    def messages(self) -> "AsyncIOMotorCollection[dict[str, Any]]":
        """Get messages collection."""
        return self._collection("messages")

    @property
    def topics(self) -> "AsyncIOMotorCollection[dict[str, Any]]":
        """Get topics collection."""
        return self._collection("topics")

    @property
    def evidence(self) -> "AsyncIOMotorCollection[dict[str, Any]]":
        """Get topic_evidence collection."""
        return self._collection("topic_evidence")

    @property
    def recall_states(self) -> "AsyncIOMotorCollection[dict[str, Any]]":
        """Get recall_states collection."""
        return self._collection("recall_states")

    async def create_indexes(self) -> None:
        """Create indexes for all collections."""
        # Chats indexes
        await self.chats.create_index("id", unique=True)
        await self.chats.create_index("source")
        await self.chats.create_index("created_on")

        # Messages indexes
        await self.messages.create_index("message_id", unique=True)
        await self.messages.create_index("chat_id")
        await self.messages.create_index("created_on")

        # Topics indexes
        await self.topics.create_index("topic_id", unique=True)
        await self.topics.create_index("canonical_name")

        # Evidence indexes
        await self.evidence.create_index("evidence_id", unique=True)
        await self.evidence.create_index("topic_id")
        await self.evidence.create_index("source_message_id")

        # Recall states indexes
        await self.recall_states.create_index("topic_id", unique=True)
        await self.recall_states.create_index("strength")

        logger.info("created_mongodb_indexes")

    async def create_vector_search_index(
        self,
        index_name: str,
        dimensions: int,
    ) -> bool:
        """Create vector search index for topics collection.

        This creates a MongoDB Atlas Vector Search index on the
        centroid_embedding field. The index must be created on Atlas
        (not available on standalone MongoDB).

        Args:
            index_name: Name for the vector search index
            dimensions: Number of dimensions in the embedding vectors

        Returns:
            True if index was created or already exists, False if not supported
        """
        try:
            # Check if vector search index already exists
            existing_indexes = await self.topics.list_search_indexes().to_list()
            for idx in existing_indexes:
                if idx.get("name") == index_name:
                    logger.info(
                        "vector_search_index_exists",
                        index_name=index_name,
                    )
                    return True

            # Create the vector search index
            # This uses the Atlas Search index creation API
            search_index_model = {
                "name": index_name,
                "type": "vectorSearch",
                "definition": {
                    "fields": [
                        {
                            "type": "vector",
                            "path": "centroid_embedding",
                            "numDimensions": dimensions,
                            "similarity": "cosine",
                        }
                    ]
                },
            }

            await self.topics.create_search_index(search_index_model)
            logger.info(
                "created_vector_search_index",
                index_name=index_name,
                dimensions=dimensions,
            )
            return True

        except Exception as e:
            # Vector search indexes are only available on MongoDB Atlas
            # For local MongoDB or older versions, this will fail
            logger.warning(
                "vector_search_index_not_available",
                error=str(e),
                hint="Vector search requires MongoDB Atlas. Falling back to in-memory search.",
            )
            return False

    @property
    def vector_search_available(self) -> bool:
        """Check if vector search is potentially available.

        This is a quick check based on URI - actual availability
        depends on Atlas configuration.
        """
        uri = self._settings.uri.get_secret_value()
        return "mongodb+srv://" in uri or "mongodb.net" in uri

    async def __aenter__(self) -> "MongoClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()
