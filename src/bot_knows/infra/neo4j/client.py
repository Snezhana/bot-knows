"""Neo4j client for bot_knows.

This module provides an async Neo4j client wrapper.
"""

from typing import Any

from neo4j import AsyncDriver, AsyncGraphDatabase

from bot_knows.config import Neo4jSettings
from bot_knows.logging import get_logger

__all__ = [
    "Neo4jClient",
]

logger = get_logger(__name__)


class Neo4jClient:
    """Async Neo4j client wrapper.

    Provides connection management and query execution
    for the bot_knows knowledge graph.

    Example:
        client = Neo4jClient(settings)
        await client.connect()

        result = await client.execute_query(
            "MATCH (n:Chat) RETURN n LIMIT 10"
        )

        await client.disconnect()
    """

    def __init__(self, settings: Neo4jSettings):
        """Initialize client with settings.

        Args:
            settings: Neo4j connection settings
        """
        self._settings = settings
        self._driver: AsyncDriver | None = None

    async def connect(self) -> None:
        """Initialize connection to Neo4j."""
        if self._driver is not None:
            return

        self._driver = AsyncGraphDatabase.driver(
            self._settings.uri,
            auth=(
                self._settings.username,
                self._settings.password.get_secret_value(),
            ),
        )

        # Verify connection
        await self._driver.verify_connectivity()
        logger.info(
            "connected_to_neo4j",
            uri=self._settings.uri,
            database=self._settings.database,
        )

    async def disconnect(self) -> None:
        """Close connection to Neo4j."""
        if self._driver:
            await self._driver.close()
            self._driver = None
            logger.info("disconnected_from_neo4j")

    @property
    def driver(self) -> AsyncDriver:
        """Get driver instance.

        Raises:
            RuntimeError: If not connected
        """
        if self._driver is None:
            raise RuntimeError("Neo4jClient not connected. Call connect() first.")
        return self._driver

    async def execute_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return results.

        Args:
            query: Cypher query string
            parameters: Query parameters

        Returns:
            List of result records as dicts
        """
        async with self.driver.session(database=self._settings.database) as session:
            result = await session.run(query, parameters or {})
            records = await result.data()
            return records

    async def execute_write(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Execute a write query (CREATE, MERGE, etc.).

        Args:
            query: Cypher query string
            parameters: Query parameters
        """
        async with self.driver.session(database=self._settings.database) as session:
            await session.run(query, parameters or {})

    async def create_indexes(self) -> None:
        """Create indexes for the knowledge graph."""
        indexes = [
            "CREATE INDEX chat_id_idx IF NOT EXISTS FOR (c:Chat) ON (c.id)",
            "CREATE INDEX message_id_idx IF NOT EXISTS FOR (m:Message) ON (m.message_id)",
            "CREATE INDEX message_chat_idx IF NOT EXISTS FOR (m:Message) ON (m.chat_id)",
            "CREATE INDEX topic_id_idx IF NOT EXISTS FOR (t:Topic) ON (t.topic_id)",
        ]

        for index_query in indexes:
            await self.execute_write(index_query)

        logger.info("created_neo4j_indexes")

    async def create_constraints(self) -> None:
        """Create uniqueness constraints."""
        constraints = [
            "CREATE CONSTRAINT chat_id_unique IF NOT EXISTS FOR (c:Chat) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT message_id_unique IF NOT EXISTS FOR (m:Message) REQUIRE m.message_id IS UNIQUE",
            "CREATE CONSTRAINT topic_id_unique IF NOT EXISTS FOR (t:Topic) REQUIRE t.topic_id IS UNIQUE",
        ]

        for constraint_query in constraints:
            try:
                await self.execute_write(constraint_query)
            except Exception as e:
                # Constraint may already exist
                logger.debug("constraint_creation_skipped", error=str(e))

        logger.info("created_neo4j_constraints")

    async def __aenter__(self) -> "Neo4jClient":
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Async context manager exit."""
        await self.disconnect()
