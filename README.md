# bot-knows

A framework-agnostic Python library for building graph-backed personal knowledge bases from chat data.
Implemented with Claude Code (model: claude-opus-4-5).

## Features

- **Multi-source Chat Ingestion**: Import chats from ChatGPT, Claude, and custom JSON formats
- **Semantic Topic Extraction**: LLM-powered topic extraction with confidence scores
- **Intelligent Deduplication**: Embedding-based semantic deduplication with configurable thresholds
- **Graph-backed Knowledge Base**: Neo4j-powered relationship graph for topics and messages
- **Evidence-weighted Recall**: Spaced repetition-inspired recall system with decay and reinforcement
- **Pluggable Infrastructure**: Bring your own storage, graph database, or LLM provider

## Requirements

- Python >= 3.13
- MongoDB (storage) - or custom storage implementation
- Neo4j (graph database) - or custom graph implementation
- Redis (optional, for caching)
- OpenAI or Anthropic API key (for LLM features) - or custom LLM implementation

## Installation

```bash
pip install bot-knows
```

Or with uv:

```bash
uv add bot-knows
```

## Quick Start

The `BotKnows` class is the main orchestrator that accepts implementation classes for storage, graph database, and LLM providers. Configuration is automatically loaded from environment variables.

### Using Built-in Infrastructure

```python
from bot_knows import (
    BotKnows,
    MongoStorageRepository,
    Neo4jGraphRepository,
    OpenAIProvider,
    ChatGPTAdapter,
)

async def main():
    # Config is loaded from .env automatically
    async with BotKnows(
        storage_class=MongoStorageRepository,
        graphdb_class=Neo4jGraphRepository,
        llm_class=OpenAIProvider,
    ) as bk:
        # Import ChatGPT conversations
        result = await bk.insert_chats("conversations.json", ChatGPTAdapter)
        print(f"Imported {result.chats_new} chats, {result.topics_created} topics")

        # Query the knowledge base
        topics = await bk.get_chat_topics(chat_id)
        due_topics = await bk.get_due_topics(threshold=0.3)
```

### Available Implementations

**Storage:**
- `MongoStorageRepository` - MongoDB-based storage

**Graph Database:**
- `Neo4jGraphRepository` - Neo4j graph database

**LLM Providers:**
- `OpenAIProvider` - OpenAI API (GPT models + embeddings)
- `AnthropicProvider` - Anthropic API (Claude models)

**Import Adapters:**
- `ChatGPTAdapter` - ChatGPT export format
- `ClaudeAdapter` - Claude export format
- `GenericJSONAdapter` - Custom JSON format


## Custom Implementations

You can provide your own implementations by implementing the required interfaces. Set `config_class = None` on your class and pass configuration via the `*_custom_config` parameters.

### Interfaces

- `StorageInterface` - Persistent storage for chats, messages, topics, evidence, and recall state
- `GraphServiceInterface` - Graph database operations for the knowledge graph
- `LLMInterface` - LLM interactions for classification and topic extraction
- `EmbeddingServiceInterface` - Text embedding generation

### Example: Custom Storage Implementation

```python
from bot_knows import BotKnows, StorageInterface, Neo4jGraphRepository, OpenAIProvider

class MyCustomStorage:
    """Custom storage implementation."""

    config_class = None  # Signals custom config

    @classmethod
    async def from_dict(cls, config: dict) -> "MyCustomStorage":
        """Factory method for custom config."""
        return cls(connection_string=config["connection_string"])

    def __init__(self, connection_string: str):
        self.conn = connection_string

    # Implement all StorageInterface methods...
    async def save_chat(self, chat): ...
    async def get_chat(self, chat_id): ...
    # ... etc

async with BotKnows(
    storage_class=MyCustomStorage,
    graphdb_class=Neo4jGraphRepository,
    llm_class=OpenAIProvider,
    storage_custom_config={"connection_string": "postgresql://..."},
) as bk:
    result = await bk.insert_chats("data.json", ChatGPTAdapter)
```

### Example: Custom LLM Provider

```python
from bot_knows import BotKnows, LLMInterface, MongoStorageRepository, Neo4jGraphRepository

class MyLLMProvider:
    """Custom LLM provider (e.g., local model, different API)."""

    config_class = None

    @classmethod
    async def from_dict(cls, config: dict) -> "MyLLMProvider":
        return cls(api_url=config["api_url"], model=config["model"])

    def __init__(self, api_url: str, model: str):
        self.api_url = api_url
        self.model = model

    # Implement LLMInterface methods
    async def classify_chat(self, first_pair, last_pair): ...
    async def extract_topics(self, user_content, assistant_content): ...
    async def normalize_topic_name(self, name): ...

    # Implement EmbeddingServiceInterface if used as embedding provider
    async def embed(self, texts): ...

async with BotKnows(
    storage_class=MongoStorageRepository,
    graphdb_class=Neo4jGraphRepository,
    llm_class=MyLLMProvider,
    llm_custom_config={"api_url": "http://localhost:8000", "model": "llama3"},
) as bk:
    result = await bk.insert_chats("data.json", ChatGPTAdapter)
```

## Configuration

Configuration is loaded from environment variables. See `.env.example` for all available options.

Key environment variables:
- `MONGODB_URI` - MongoDB connection string
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD` - Neo4j connection
- `OPENAI_API_KEY` - OpenAI API key
- `ANTHROPIC_API_KEY` - Anthropic API key
- `DEDUP_HIGH_THRESHOLD`, `DEDUP_LOW_THRESHOLD` - Deduplication thresholds

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

## Retrieval API

```python
async with BotKnows(...) as bk:
    # Get messages for a chat
    messages = await bk.get_messages_for_chat(chat_id)

    # Get topics for a chat
    topic_ids = await bk.get_chat_topics(chat_id)

    # Get related topics
    related = await bk.get_related_topics(topic_id, limit=10)

    # Get topic evidence
    evidence = await bk.get_topic_evidence(topic_id)

    # Spaced repetition recall
    recall_state = await bk.get_recall_state(topic_id)
    due_topics = await bk.get_due_topics(threshold=0.3)
    all_states = await bk.get_all_recall_states()
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

## Future Plans

The built-in infrastructure will be extended with additional providers:

- **Storage**: PostgreSQL, SQLite
- **Graph**: Amazon Neptune, TigerGraph, MemGraph
- **LLM**: Google Gemini, Ollama, HuggingFace

## Contributing

Contributions are welcome! If you'd like to add a new infrastructure implementation:

1. Implement the appropriate interface (`StorageInterface`, `GraphServiceInterface`, `LLMInterface`, or `EmbeddingServiceInterface`)
2. Add a `config_class` for environment-based configuration (or set to `None` for custom config)
3. Implement the `from_config` class method (or `from_dict` if `config_class` is `None`)
4. Add tests for your implementation
5. Submit a pull request

## License

MIT
