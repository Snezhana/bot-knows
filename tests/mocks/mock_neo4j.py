"""Mock Neo4j client for testing."""

from typing import Any


class MockNeo4jClient:
    """Mock Neo4j client for testing."""

    def __init__(self) -> None:
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: list[dict[str, Any]] = []

    async def connect(self) -> None:
        pass

    async def disconnect(self) -> None:
        pass

    async def execute_query(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        # Return empty list for queries
        return []

    async def execute_write(
        self,
        query: str,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        # Store node/edge data based on query
        params = parameters or {}

        if "MERGE (c:Chat" in query:
            self._nodes[f"chat:{params.get('id')}"] = {
                "type": "Chat",
                **params,
            }
        elif "MERGE (m:Message" in query:
            self._nodes[f"message:{params.get('message_id')}"] = {
                "type": "Message",
                **params,
            }
        elif "MERGE (t:Topic" in query:
            self._nodes[f"topic:{params.get('topic_id')}"] = {
                "type": "Topic",
                **params,
            }
        elif "IS_PART_OF" in query:
            self._edges.append({
                "type": "IS_PART_OF",
                "from": params.get("message_id"),
                "to": params.get("chat_id"),
            })
        elif "FOLLOWS_AFTER" in query:
            self._edges.append({
                "type": "FOLLOWS_AFTER",
                "from": params.get("message_id"),
                "to": params.get("previous_message_id"),
            })
        elif "IS_SUPPORTED_BY" in query:
            self._edges.append({
                "type": "IS_SUPPORTED_BY",
                "from": params.get("topic_id"),
                "to": params.get("message_id"),
                **{k: v for k, v in params.items() if k not in ("topic_id", "message_id")},
            })

    async def create_indexes(self) -> None:
        pass

    async def create_constraints(self) -> None:
        pass

    # Helper methods for tests
    def get_node(self, node_type: str, node_id: str) -> dict[str, Any] | None:
        return self._nodes.get(f"{node_type}:{node_id}")

    def get_edges(self, edge_type: str) -> list[dict[str, Any]]:
        return [e for e in self._edges if e["type"] == edge_type]
