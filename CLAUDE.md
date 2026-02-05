# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

bot-knows is a framework-agnostic Python library for building graph-backed personal knowledge bases from chat data. It features LLM-powered topic extraction, semantic deduplication, and spaced-repetition recall.

**Python 3.13+ required** | Package manager: `uv`

## Common Commands

```bash
# Install dependencies
uv sync --dev --all-extras

# Testing
uv run pytest                              # All tests
uv run pytest tests/unit/                  # Unit tests only
uv run pytest tests/unit/test_models.py   # Single file
uv run pytest -k "test_name"              # Single test by name
uv run pytest --cov=bot_knows             # With coverage

# Type checking & linting
uv run mypy src/
uv run ruff check src/
uv run ruff format src/

# Building
uv build
```

## Architecture

```
Chat Data (ChatGPT/Claude exports)
       ↓
Import Adapters → ChatIngest (frozen boundary model)
       ↓
Domain Processing (identity, classification, ordering)
       ↓
Topic Extraction → Semantic Deduplication → Graph Updates
       ↓
Retrieval & Recall (spaced repetition)
```

### Module Structure

- **`interfaces/`** - Protocol-based contracts (StorageInterface, GraphServiceInterface, LLMInterface, EmbeddingServiceInterface)
- **`models/`** - Public DTOs (ChatDTO, MessageDTO, TopicDTO, ChatIngest) - all frozen/immutable
- **`domain/`** - Internal entities (not public API)
- **`services/`** - Business logic (chat processing, topic extraction, deduplication, recall)
- **`infra/`** - Concrete implementations (mongo, neo4j, redis, llm providers)
- **`importers/`** - Chat import adapters (chatgpt, claude, generic_json)
- **`orchestrator.py`** - Main `BotKnows` class (async context manager)

### Key Patterns

1. **Async context manager**: Always use `async with BotKnows(...) as bk:`
2. **Protocol interfaces**: Use `Protocol` with `@runtime_checkable` for all contracts
3. **Factory pattern**: Implementations have `from_config(config)` classmethod; `config_class` attribute signals auto-load from env
4. **Immutable boundaries**: ChatIngest and all DTOs are frozen Pydantic models
5. **Append-only evidence**: Topic evidence is never mutated, only appended
6. **Three-tier deduplication**: ≥0.92 merge, 0.80-0.92 soft link, <0.80 new topic

### Graph Model

```
Nodes: Chat, Message, Topic
Edges:
  (Message)-[:IS_PART_OF]->(Chat)
  (Message)-[:FOLLOWS_AFTER]->(Message)
  (Topic)-[:IS_SUPPORTED_BY]->(Message)
```

## Configuration

Environment variables prefixed with `BOT_KNOWS_`. Key settings:
- `BOT_KNOWS_MONGO_URI`, `BOT_KNOWS_NEO4J_URI`, `BOT_KNOWS_REDIS_URL`
- `BOT_KNOWS_LLM_PROVIDER` (openai/anthropic), `BOT_KNOWS_LLM_API_KEY`
- `BOT_KNOWS_DEDUP_HIGH_THRESHOLD` (0.92), `BOT_KNOWS_DEDUP_LOW_THRESHOLD` (0.80)

See `.env.example` for full list.

## Testing

- `tests/unit/` - Fast tests, no external deps (marker: `@pytest.mark.unit`)
- `tests/integration/` - Requires Docker (marker: `@pytest.mark.integration`)
- `tests/mocks/` - Mock implementations for testing
- Fixtures in `tests/conftest.py`

## Extending

Implement the appropriate interface and add `config_class` for env-based config (or `None` for manual):
- Storage: `StorageInterface`
- Graph: `GraphServiceInterface`
- LLM: `LLMInterface`
- Embeddings: `EmbeddingServiceInterface`
- Import adapters: `ChatImportAdapter`
