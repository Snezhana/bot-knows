#!/usr/bin/env python
"""Migration script: Remove Chat nodes from Neo4j graph.

This script migrates the graph structure by:
1. Copying chat properties (title, source, category, tags) from Chat nodes to Message nodes
2. Deleting all IS_PART_OF edges
3. Deleting all Chat nodes

The script is idempotent - safe to run multiple times.

Usage:
    python scripts/migrate_graph_remove_chat_node.py

Prerequisites:
    1. Neo4j running on configured URI
    2. .env file with Neo4j credentials configured

Environment variables (via .env):
    BOT_KNOWS_NEO4J_URI=bolt://localhost:7687
    BOT_KNOWS_NEO4J_USERNAME=neo4j
    BOT_KNOWS_NEO4J_PASSWORD=your_password
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot_knows.config import BotKnowsConfig
from bot_knows.infra.neo4j.client import Neo4jClient


async def count_nodes_and_edges(client: Neo4jClient) -> dict[str, int]:
    """Count current nodes and edges for reporting."""
    counts = {}

    # Count Chat nodes
    result = await client.execute_query("MATCH (c:Chat) RETURN count(c) as count")
    counts["chat_nodes"] = result[0]["count"]

    # Count Message nodes
    result = await client.execute_query("MATCH (m:Message) RETURN count(m) as count")
    counts["message_nodes"] = result[0]["count"]

    # Count IS_PART_OF edges
    result = await client.execute_query("MATCH ()-[r:IS_PART_OF]->() RETURN count(r) as count")
    counts["is_part_of_edges"] = result[0]["count"]

    # Count Messages with chat_title property
    result = await client.execute_query(
        "MATCH (m:Message) WHERE m.chat_title IS NOT NULL RETURN count(m) as count"
    )
    counts["messages_with_chat_title"] = result[0]["count"]

    return counts


async def migrate_chat_properties_to_messages(client: Neo4jClient) -> int:
    """Copy chat properties from Chat nodes to related Message nodes.

    Returns:
        Number of messages updated
    """
    query = """
    MATCH (m:Message)-[:IS_PART_OF]->(c:Chat)
    WHERE m.chat_title IS NULL
    SET m.chat_title = c.title,
        m.source = c.source,
        m.category = c.category,
        m.tags = c.tags
    RETURN count(m) as updated
    """
    result = await client.execute_query(query)
    return result[0]["updated"] if result else 0


async def delete_is_part_of_edges(client: Neo4jClient) -> int:
    """Delete all IS_PART_OF edges.

    Returns:
        Number of edges deleted
    """
    # First count
    count_result = await client.execute_query(
        "MATCH ()-[r:IS_PART_OF]->() RETURN count(r) as count"
    )
    count = count_result[0]["count"] if count_result else 0

    if count > 0:
        await client.execute_write("MATCH ()-[r:IS_PART_OF]->() DELETE r")

    return count


async def delete_chat_nodes(client: Neo4jClient) -> int:
    """Delete all Chat nodes.

    Returns:
        Number of nodes deleted
    """
    # First count
    count_result = await client.execute_query("MATCH (c:Chat) RETURN count(c) as count")
    count = count_result[0]["count"] if count_result else 0

    if count > 0:
        await client.execute_write("MATCH (c:Chat) DELETE c")

    return count


async def drop_chat_index_and_constraint(client: Neo4jClient) -> None:
    """Drop the Chat index and constraint if they exist."""
    # Drop constraint first (constraint implies index)
    try:
        await client.execute_write("DROP CONSTRAINT chat_id_unique IF EXISTS")
        print("  Dropped constraint: chat_id_unique")
    except Exception as e:
        print(f"  Note: Could not drop constraint (may not exist): {e}")

    # Drop index
    try:
        await client.execute_write("DROP INDEX chat_id_idx IF EXISTS")
        print("  Dropped index: chat_id_idx")
    except Exception as e:
        print(f"  Note: Could not drop index (may not exist): {e}")


async def main() -> None:
    """Run the migration."""
    print("\n" + "=" * 60)
    print("Migration: Remove Chat Nodes from Graph")
    print("=" * 60)

    # Load configuration
    config = BotKnowsConfig()

    print(f"\nConnecting to Neo4j: {config.neo4j.uri}...")
    client = Neo4jClient(config.neo4j)
    await client.connect()
    print("  Connected successfully")

    try:
        # Step 1: Count current state
        print("\n--- Current Graph State ---")
        before_counts = await count_nodes_and_edges(client)
        print(f"  Chat nodes: {before_counts['chat_nodes']}")
        print(f"  Message nodes: {before_counts['message_nodes']}")
        print(f"  IS_PART_OF edges: {before_counts['is_part_of_edges']}")
        print(f"  Messages with chat_title: {before_counts['messages_with_chat_title']}")

        # Check if migration is needed
        if before_counts["chat_nodes"] == 0 and before_counts["is_part_of_edges"] == 0:
            print("\n--- Migration Already Complete ---")
            print("  No Chat nodes or IS_PART_OF edges found.")
            print("  Graph is already in the new structure.")
            return

        # Step 2: Migrate chat properties to messages
        print("\n--- Step 1: Migrating Chat Properties to Messages ---")
        updated = await migrate_chat_properties_to_messages(client)
        print(f"  Updated {updated} message(s) with chat properties")

        # Step 3: Delete IS_PART_OF edges
        print("\n--- Step 2: Deleting IS_PART_OF Edges ---")
        deleted_edges = await delete_is_part_of_edges(client)
        print(f"  Deleted {deleted_edges} IS_PART_OF edge(s)")

        # Step 4: Delete Chat nodes
        print("\n--- Step 3: Deleting Chat Nodes ---")
        deleted_nodes = await delete_chat_nodes(client)
        print(f"  Deleted {deleted_nodes} Chat node(s)")

        # Step 5: Drop Chat index and constraint
        print("\n--- Step 4: Dropping Chat Index and Constraint ---")
        await drop_chat_index_and_constraint(client)

        # Step 6: Verify final state
        print("\n--- Final Graph State ---")
        after_counts = await count_nodes_and_edges(client)
        print(f"  Chat nodes: {after_counts['chat_nodes']}")
        print(f"  Message nodes: {after_counts['message_nodes']}")
        print(f"  IS_PART_OF edges: {after_counts['is_part_of_edges']}")
        print(f"  Messages with chat_title: {after_counts['messages_with_chat_title']}")

        # Summary
        print("\n" + "=" * 60)
        print("MIGRATION COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print("\nSummary:")
        print(f"  - Migrated {updated} messages with chat properties")
        print(f"  - Deleted {deleted_edges} IS_PART_OF edges")
        print(f"  - Deleted {deleted_nodes} Chat nodes")
        print("  - Dropped Chat index and constraint")

    finally:
        await client.disconnect()
        print("\nDisconnected from Neo4j")


if __name__ == "__main__":
    asyncio.run(main())
