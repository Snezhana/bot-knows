#!/usr/bin/env python
"""End-to-end test script for bot_knows.

This script tests the full pipeline with real MongoDB, Neo4j, and LLM API.
Redis is disabled for this test.

Usage:
    python scripts/e2e_test.py

Prerequisites:
    1. MongoDB running on configured URI
    2. Neo4j running on configured URI
    3. .env file with LLM API key configured

Environment variables (via .env):
    BOT_KNOWS_MONGO_URI=mongodb://localhost:27017
    BOT_KNOWS_MONGO_DATABASE=bot_knows_test
    BOT_KNOWS_NEO4J_URI=bolt://localhost:7687
    BOT_KNOWS_NEO4J_USERNAME=neo4j
    BOT_KNOWS_NEO4J_PASSWORD=your_password
    BOT_KNOWS_LLM_PROVIDER=openai
    BOT_KNOWS_LLM_API_KEY=your_api_key
    BOT_KNOWS_LLM_MODEL=gpt-4o
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot_knows.config import BotKnowsConfig
from bot_knows.importers.chatgpt import ChatGPTAdapter
from bot_knows.importers.claude import ClaudeAdapter
from bot_knows.infra.llm.anthropic_provider import AnthropicProvider
from bot_knows.infra.llm.openai_provider import OpenAIProvider
from bot_knows.infra.mongo.client import MongoClient
from bot_knows.infra.mongo.repositories import MongoStorageRepository
from bot_knows.infra.neo4j.client import Neo4jClient
from bot_knows.infra.neo4j.graph_repository import Neo4jGraphRepository
from bot_knows.logging import configure_logging, get_logger
from bot_knows.services.chat_processing import ChatProcessingService
from bot_knows.services.dedup_service import DedupAction, DedupService
from bot_knows.services.graph_service import GraphService
from bot_knows.services.message_builder import MessageBuilder
from bot_knows.services.recall_service import RecallService
from bot_knows.services.topic_extraction import TopicExtractionService

# Configure logging
configure_logging(level=logging.INFO)
logger = get_logger(__name__)


class E2ETestRunner:
    """End-to-end test runner for bot_knows pipeline."""

    def __init__(self, config: BotKnowsConfig) -> None:
        """Initialize test runner with configuration."""
        self.config = config
        self.mongo_client: MongoClient | None = None
        self.neo4j_client: Neo4jClient | None = None
        self.storage: MongoStorageRepository | None = None
        self.graph_repo: Neo4jGraphRepository | None = None
        self.llm_provider: OpenAIProvider | AnthropicProvider | None = None
        self.embedding_provider: OpenAIProvider | None = None

    async def setup(self) -> None:
        """Set up infrastructure connections."""
        print("\n" + "=" * 60)
        print("Setting up infrastructure connections...")
        print("=" * 60)

        # MongoDB
        print(f"\nConnecting to MongoDB: {self.config.mongo.database}...")
        self.mongo_client = MongoClient(self.config.mongo)
        await self.mongo_client.connect()
        await self.mongo_client.create_indexes()
        self.storage = MongoStorageRepository(self.mongo_client)
        print("  MongoDB connected successfully")

        # Neo4j
        print(f"\nConnecting to Neo4j: {self.config.neo4j.uri}...")
        self.neo4j_client = Neo4jClient(self.config.neo4j)
        await self.neo4j_client.connect()
        await self.neo4j_client.create_indexes()
        await self.neo4j_client.create_constraints()
        self.graph_repo = Neo4jGraphRepository(self.neo4j_client)
        print("  Neo4j connected successfully")

        # LLM Provider
        print(f"\nInitializing LLM provider: {self.config.llm.provider}...")
        if self.config.llm.provider == "openai":
            self.llm_provider = OpenAIProvider(self.config.llm)
            self.embedding_provider = self.llm_provider  # OpenAI provides both
            print(f"  OpenAI provider initialized (model: {self.config.llm.model})")
        elif self.config.llm.provider == "anthropic":
            self.llm_provider = AnthropicProvider(self.config.llm)
            # Anthropic doesn't provide embeddings, use OpenAI for embeddings
            print(f"  Anthropic provider initialized (model: {self.config.llm.model})")
            print("  Note: Using OpenAI for embeddings (Anthropic doesn't provide embeddings)")
            # Create a separate OpenAI provider for embeddings
            from pydantic import SecretStr

            from bot_knows.config import LLMSettings

            openai_key = os.environ.get("OPENAI_API_KEY") or os.environ.get(
                "BOT_KNOWS_OPENAI_API_KEY"
            )
            if not openai_key:
                raise ValueError(
                    "When using Anthropic, you need OPENAI_API_KEY or BOT_KNOWS_OPENAI_API_KEY "
                    "environment variable set for embeddings"
                )
            embedding_settings = LLMSettings(
                provider="openai",
                api_key=SecretStr(openai_key),
                embedding_model=self.config.llm.embedding_model,
            )
            self.embedding_provider = OpenAIProvider(embedding_settings)
            print(
                f"OpenAI embedding provider initialized (model: {self.config.llm.embedding_model})"
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.config.llm.provider}")

    async def teardown(self) -> None:
        """Clean up infrastructure connections."""
        print("\n" + "=" * 60)
        print("Cleaning up connections...")
        print("=" * 60)

        if self.neo4j_client:
            await self.neo4j_client.disconnect()
            print("  Neo4j disconnected")

        if self.mongo_client:
            await self.mongo_client.disconnect()
            print("  MongoDB disconnected")

    async def clear_test_data(self) -> None:
        """Clear any existing test data."""
        print("\nClearing existing test data...")

        if self.mongo_client:
            await self.mongo_client.chats.delete_many({})
            await self.mongo_client.messages.delete_many({})
            await self.mongo_client.topics.delete_many({})
            await self.mongo_client.evidence.delete_many({})
            await self.mongo_client.recall_states.delete_many({})
            print("  MongoDB collections cleared")

        if self.neo4j_client:
            # Clear all nodes and relationships
            await self.neo4j_client.execute_write("MATCH (n) DETACH DELETE n")
            print("  Neo4j graph cleared")

    async def test_chatgpt_import(self) -> dict:
        """Test ChatGPT import pipeline."""
        print("\n" + "=" * 60)
        print("TEST 1: ChatGPT Import Pipeline")
        print("=" * 60)

        results = {
            "chats_imported": 0,
            "messages_created": 0,
            "topics_extracted": 0,
            "topics_merged": 0,
            "topics_new": 0,
        }

        # Load fixture
        fixture_path = Path(__file__).parent.parent / "tests" / "fixtures" / "chatgpt_export.json"
        print(f"\nLoading fixture: {fixture_path.name}")

        with open(fixture_path, encoding="utf-8") as f:
            raw_export = json.load(f)

        # Parse with adapter
        adapter = ChatGPTAdapter()
        chat_ingests = adapter.parse(raw_export)
        print(f"  Parsed {len(chat_ingests)} chat(s) from export")

        # Initialize services
        chat_service = ChatProcessingService(self.storage, self.llm_provider)
        message_builder = MessageBuilder()
        topic_extractor = TopicExtractionService(self.llm_provider, self.embedding_provider)
        dedup_service = DedupService(
            self.embedding_provider,
            self.storage,
            high_threshold=self.config.dedup_high_threshold,
            low_threshold=self.config.dedup_low_threshold,
        )
        graph_service = GraphService(self.graph_repo)

        # Process each chat
        for chat_ingest in chat_ingests:
            print(f"\nProcessing chat: {chat_ingest.title}")

            # Step 1: Process chat (classification)
            print("  Step 1: Chat classification...")
            chat_dto, is_new = await chat_service.process(chat_ingest)
            print(f"    Category: {chat_dto.category.value}")
            print(f"    Tags: {chat_dto.tags}")
            print(f"    Is new: {is_new}")
            results["chats_imported"] += 1 if is_new else 0

            # Step 2: Build messages
            print("  Step 2: Building messages...")
            messages = message_builder.build(chat_ingest.messages, chat_dto.id)
            print(f"    Created {len(messages)} message pair(s)")

            # Save messages to storage
            for msg in messages:
                await self.storage.save_message(msg)
            results["messages_created"] += len(messages)

            # Step 3: Add to graph
            print("  Step 3: Adding to graph...")
            await graph_service.add_chat_with_messages(chat_dto, messages)
            print("    Chat and messages added to Neo4j")

            # Step 4: Extract topics from each message
            print("  Step 4: Topic extraction...")
            for msg in messages:
                if msg.is_empty:
                    continue

                candidates = await topic_extractor.extract(msg)
                print(f"    Message {msg.message_id[:8]}... -> {len(candidates)} topic(s)")

                for candidate in candidates:
                    results["topics_extracted"] += 1

                    # Step 5: Deduplication
                    dedup_result = await dedup_service.check_duplicate(candidate.embedding)

                    if dedup_result.action == DedupAction.MERGE:
                        # Merge with existing topic
                        print(
                            f"""'{candidate.extracted_name}' -> MERGE with existing
                            (sim: {dedup_result.similarity:.3f})"""
                        )
                        topic, evidence = await topic_extractor.create_evidence_for_existing(
                            candidate, dedup_result.existing_topic
                        )
                        await self.storage.update_topic(topic)
                        await self.storage.append_evidence(evidence)
                        await graph_service.add_evidence_to_existing_topic(topic, evidence)
                        results["topics_merged"] += 1

                    elif dedup_result.action == DedupAction.SOFT_MATCH:
                        # Create new with potential duplicate link
                        print(
                            f"""      '{candidate.extracted_name}' -> NEW + link
                            (sim: {dedup_result.similarity:.3f})"""
                        )
                        topic, evidence = await topic_extractor.create_topic_and_evidence(candidate)
                        await self.storage.save_topic(topic)
                        await self.storage.append_evidence(evidence)
                        await graph_service.add_topic_with_evidence(topic, evidence)
                        await graph_service.create_potential_duplicate_link(
                            topic.topic_id,
                            dedup_result.existing_topic.topic_id,
                            dedup_result.similarity,
                        )
                        results["topics_new"] += 1

                    else:
                        # Completely new topic
                        print(f"      '{candidate.extracted_name}' -> NEW")
                        topic, evidence = await topic_extractor.create_topic_and_evidence(candidate)
                        await self.storage.save_topic(topic)
                        await self.storage.append_evidence(evidence)
                        await graph_service.add_topic_with_evidence(topic, evidence)
                        results["topics_new"] += 1

        return results

    async def test_claude_import(self) -> dict:
        """Test Claude import pipeline."""
        print("\n" + "=" * 60)
        print("TEST 2: Claude Import Pipeline")
        print("=" * 60)

        results = {
            "chats_imported": 0,
            "messages_created": 0,
            "topics_extracted": 0,
        }

        # Load fixture
        fixture_path = Path(__file__).parent.parent / "tests" / "fixtures" / "claude_export.json"
        print(f"\nLoading fixture: {fixture_path.name}")

        with open(fixture_path, encoding="utf-8") as f:
            raw_export = json.load(f)

        # Parse with adapter
        adapter = ClaudeAdapter()
        chat_ingests = adapter.parse(raw_export)
        print(f"  Parsed {len(chat_ingests)} chat(s) from export")

        # Initialize services
        chat_service = ChatProcessingService(self.storage, self.llm_provider)
        message_builder = MessageBuilder()
        topic_extractor = TopicExtractionService(self.llm_provider, self.embedding_provider)
        dedup_service = DedupService(
            self.embedding_provider,
            self.storage,
            high_threshold=self.config.dedup_high_threshold,
            low_threshold=self.config.dedup_low_threshold,
        )
        graph_service = GraphService(self.graph_repo)

        for chat_ingest in chat_ingests:
            print(f"\nProcessing chat: {chat_ingest.title}")

            # Process chat
            chat_dto, is_new = await chat_service.process(chat_ingest)
            print(f"    Category: {chat_dto.category.value}")
            print(f"    Tags: {chat_dto.tags}")
            results["chats_imported"] += 1 if is_new else 0

            # Build and save messages
            messages = message_builder.build(chat_ingest.messages, chat_dto.id)
            for msg in messages:
                await self.storage.save_message(msg)
            results["messages_created"] += len(messages)

            # Add to graph
            await graph_service.add_chat_with_messages(chat_dto, messages)

            # Extract topics
            for msg in messages:
                if msg.is_empty:
                    continue

                candidates = await topic_extractor.extract(msg)
                results["topics_extracted"] += len(candidates)

                for candidate in candidates:
                    dedup_result = await dedup_service.check_duplicate(candidate.embedding)

                    if dedup_result.action == DedupAction.MERGE:
                        topic, evidence = await topic_extractor.create_evidence_for_existing(
                            candidate, dedup_result.existing_topic
                        )
                        await self.storage.update_topic(topic)
                        await self.storage.append_evidence(evidence)
                        await graph_service.add_evidence_to_existing_topic(topic, evidence)
                    elif dedup_result.action == DedupAction.SOFT_MATCH:
                        topic, evidence = await topic_extractor.create_topic_and_evidence(candidate)
                        await self.storage.save_topic(topic)
                        await self.storage.append_evidence(evidence)
                        await graph_service.add_topic_with_evidence(topic, evidence)
                        await graph_service.create_potential_duplicate_link(
                            topic.topic_id,
                            dedup_result.existing_topic.topic_id,
                            dedup_result.similarity,
                        )
                    else:
                        topic, evidence = await topic_extractor.create_topic_and_evidence(candidate)
                        await self.storage.save_topic(topic)
                        await self.storage.append_evidence(evidence)
                        await graph_service.add_topic_with_evidence(topic, evidence)

        return results

    async def test_recall_service(self) -> dict:
        """Test recall reinforcement and decay."""
        print("\n" + "=" * 60)
        print("TEST 3: Recall Service")
        print("=" * 60)

        results = {
            "topics_reinforced": 0,
            "due_topics": 0,
        }

        recall_service = RecallService(
            self.storage,
            self.graph_repo,
            stability_k=self.config.recall_stability_k,
            semantic_boost=self.config.recall_semantic_boost,
        )

        # Get all topics
        all_topics = await self.storage.get_all_topics()
        print(f"\nFound {len(all_topics)} topic(s) in database")

        # Reinforce each topic
        print("\nReinforcing topics with 'active' context...")
        for topic in all_topics[:5]:  # Limit to first 5 for demo
            state = await recall_service.reinforce(
                topic_id=topic.topic_id,
                confidence=0.8,
                novelty_factor=1.0,
                context="active",
            )
            print(
                f"""  {topic.canonical_name}: strength={state.strength:.3f},
                stability={state.stability:.3f}"""
            )
            results["topics_reinforced"] += 1

        # Get due topics
        print("\nChecking for due topics (threshold=0.3)...")
        due_topics = await recall_service.get_due_topics(threshold=0.3, limit=10)
        results["due_topics"] = len(due_topics)

        for item in due_topics[:3]:  # Show first 3
            print(f"  {item.topic.canonical_name}:")
            print(f"    strength={item.recall_state.strength:.3f}")
            print(f"    due_score={item.due_score:.3f}")
            print(f"    related={item.related_topics[:3]}")

        return results

    async def verify_graph_structure(self) -> dict:
        """Verify the Neo4j graph structure."""
        print("\n" + "=" * 60)
        print("TEST 4: Graph Structure Verification")
        print("=" * 60)

        results = {}

        # Count nodes
        chat_count = await self.neo4j_client.execute_query(
            "MATCH (c:Chat) RETURN count(c) as count"
        )
        results["chat_nodes"] = chat_count[0]["count"]

        message_count = await self.neo4j_client.execute_query(
            "MATCH (m:Message) RETURN count(m) as count"
        )
        results["message_nodes"] = message_count[0]["count"]

        topic_count = await self.neo4j_client.execute_query(
            "MATCH (t:Topic) RETURN count(t) as count"
        )
        results["topic_nodes"] = topic_count[0]["count"]

        print("\nNode counts:")
        print(f"  Chat nodes: {results['chat_nodes']}")
        print(f"  Message nodes: {results['message_nodes']}")
        print(f"  Topic nodes: {results['topic_nodes']}")

        # Count relationships
        is_part_of = await self.neo4j_client.execute_query(
            "MATCH ()-[r:IS_PART_OF]->() RETURN count(r) as count"
        )
        results["is_part_of_edges"] = is_part_of[0]["count"]

        follows_after = await self.neo4j_client.execute_query(
            "MATCH ()-[r:FOLLOWS_AFTER]->() RETURN count(r) as count"
        )
        results["follows_after_edges"] = follows_after[0]["count"]

        is_supported_by = await self.neo4j_client.execute_query(
            "MATCH ()-[r:IS_SUPPORTED_BY]->() RETURN count(r) as count"
        )
        results["is_supported_by_edges"] = is_supported_by[0]["count"]

        potentially_duplicate = await self.neo4j_client.execute_query(
            "MATCH ()-[r:POTENTIALLY_DUPLICATE_OF]->() RETURN count(r) as count"
        )
        results["potentially_duplicate_edges"] = potentially_duplicate[0]["count"]

        print("\nRelationship counts:")
        print(f"  IS_PART_OF: {results['is_part_of_edges']}")
        print(f"  FOLLOWS_AFTER: {results['follows_after_edges']}")
        print(f"  IS_SUPPORTED_BY: {results['is_supported_by_edges']}")
        print(f"  POTENTIALLY_DUPLICATE_OF: {results['potentially_duplicate_edges']}")

        # Sample query: Get topics for a chat
        sample_query = await self.neo4j_client.execute_query("""
            MATCH (t:Topic)-[:IS_SUPPORTED_BY]->(m:Message)-[:IS_PART_OF]->(c:Chat)
            RETURN c.title as chat, collect(DISTINCT t.canonical_name) as topics
            LIMIT 3
        """)

        print("\nSample: Topics per chat:")
        for row in sample_query:
            topics = row["topics"][:5]  # Limit display
            print(f"  '{row['chat'][:40]}...' -> {topics}")

        return results

    async def run_all_tests(self) -> None:
        """Run all E2E tests."""
        print("\n" + "#" * 60)
        print("#  bot_knows End-to-End Test Suite")
        print("#" * 60)

        try:
            await self.setup()
            await self.clear_test_data()

            # Run tests
            chatgpt_results = await self.test_chatgpt_import()
            # claude_results = await self.test_claude_import()
            recall_results = await self.test_recall_service()
            graph_results = await self.verify_graph_structure()

            # Print summary
            print("\n" + "=" * 60)
            print("TEST SUMMARY")
            print("=" * 60)

            print("\nChatGPT Import:")
            for key, value in chatgpt_results.items():
                print(f"  {key}: {value}")

            # print("\nClaude Import:")
            # for key, value in claude_results.items():
            #     print(f"  {key}: {value}")

            print("\nRecall Service:")
            for key, value in recall_results.items():
                print(f"  {key}: {value}")

            print("\nGraph Structure:")
            for key, value in graph_results.items():
                print(f"  {key}: {value}")

            print("\n" + "=" * 60)
            print("ALL TESTS COMPLETED SUCCESSFULLY")
            print("=" * 60)

        except Exception as e:
            logger.exception("E2E test failed")
            print(f"\nERROR: {e}")
            raise

        finally:
            await self.teardown()


async def main() -> None:
    """Main entry point."""
    # Load configuration from environment
    config = BotKnowsConfig()

    # Validate required config
    if not config.llm.api_key:
        print("ERROR: BOT_KNOWS_LLM_API_KEY not set in environment")
        print("Please set up your .env file with the required variables.")
        sys.exit(1)

    # Disable Redis for this test
    config.redis.enabled = False

    runner = E2ETestRunner(config)
    await runner.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
