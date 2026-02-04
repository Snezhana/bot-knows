# bot-knows

A framework-agnostic Python library for building graph-backed personal knowledge bases from chat data.

## Features

- **Multi-source Chat Ingestion**: Import chats from ChatGPT, Claude, and custom JSON formats
- **Semantic Topic Extraction**: LLM-powered topic extraction with confidence scores
- **Intelligent Deduplication**: Embedding-based semantic deduplication with configurable thresholds
- **Graph-backed Knowledge Base**: Neo4j-powered relationship graph for topics and messages
- **Evidence-weighted Recall**: Spaced repetition-inspired recall system with decay and reinforcement

## Requirements

- Python >= 3.13
- MongoDB (storage)
- Neo4j (graph database)
- Redis (optional, for caching)
- OpenAI or Anthropic API key (for LLM features)

## Installation

```bash
pip install bot-knows
```

Or with uv:

```bash
uv add bot-knows
```

## Quick Start

```python
from bot_knows import (
    ChatProcessingService,
    ImportAdapterRegistry,
    BotKnowsConfig,
)

# Load configuration from environment
config = BotKnowsConfig()

# Import a ChatGPT export
adapter = ImportAdapterRegistry.create("chatgpt")
chats = adapter.parse_file("path/to/conversations.json")

# Process each chat
for chat_ingest in chats:
    chat, is_new = await chat_processor.process(chat_ingest)
    if is_new:
        print(f"Imported: {chat.title}")
```

## Configuration

Configuration is loaded from environment variables. See `.env.example` for all available options.

## Architecture

```
Input Sources (ChatGPT, Claude, Custom JSON)
        ↓
Import Adapters (normalize to ChatIngest)
        ↓
Domain Processing
  ├── Chat identity resolution
  ├── One-time Chat classification
  ├── Message creation & ordering
        ↓
Topic Extraction
  ├── LLM-based extraction
  ├── Semantic deduplication
  ├── Evidence append
        ↓
Graph Updates (Neo4j)
```

## Development

```bash
# Install with dev dependencies
uv sync --dev

# Run tests
uv run pytest

# Type checking
uv run mypy src/

# Linting
uv run ruff check src/
```

## License

MIT
