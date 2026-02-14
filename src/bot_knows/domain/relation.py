"""Graph relationship types for bot_knows.

This module defines the edge types used in the Neo4j knowledge graph.
"""

from enum import StrEnum

__all__ = [
    "RelationType",
]


class RelationType(StrEnum):
    """Graph edge types for the knowledge graph.

    These define the relationships between nodes in Neo4j.
    """

    # Message relationships
    FOLLOWS_AFTER = "FOLLOWS_AFTER"
    """(Message)-[:FOLLOWS_AFTER]->(Message) - defines ordering"""

    # Topic relationships
    IS_SUPPORTED_BY = "IS_SUPPORTED_BY"
    """(Topic)-[:IS_SUPPORTED_BY {evidence}]->(Message)"""

    RELATES_TO = "RELATES_TO"
    """(Topic)-[:RELATES_TO {type, weight}]->(Topic)"""


class SemanticRelationType(StrEnum):
    """Semantic relationship types between topics.

    Used as the 'type' property on RELATES_TO edges.
    """

    PART_OF = "part_of"
    """Topic A is part of Topic B"""

    CAUSES = "causes"
    """Topic A causes Topic B"""

    RELATED_TO = "related_to"
    """General semantic relationship"""

    PREREQUISITE_OF = "prerequisite_of"
    """Topic A is a prerequisite for Topic B"""

    SIMILAR_TO = "similar_to"
    """Topics are semantically similar"""
