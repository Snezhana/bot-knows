# bot_knows Python Package
## Developer Implementation Guide (Updated Design)

This document defines the **complete design and public contract** of the `bot_knows` pip package.

The goal: **developers can implement the core package without ambiguity**.

---

## 1. Purpose of the Package

`bot_knows` is a **framework-agnostic Python library** that:
- Ingests chats from many sources (exports, live capture, scripts)
- Groups imported data into Chats and Messages
- Classifies Chats (one-time)
- Extracts reusable Topics from Messages
- Builds a graph-backed personal knowledge base

It intentionally **does not** include:
- HTTP APIs
- UI concerns
- Provider-specific UX logic

---

## 2. High-level Architecture

```
Input Sources
  ├── Chat Exports (ChatGPT, Claude, …)
  ├── Live Messages
  └── Custom JSON
        ↓
Import Adapters
        ↓
ChatIngest (frozen, importer output)
        ↓
Domain Processing
  ├── Chat identity resolution
  ├── One-time Chat classification
  ├── Chat persistence (metadata only)
  ├── Message creation & ordering
        ↓
Topic Extraction
  ├── Candidate Topic extraction
  ├── Semantic deduplication
  ├── Evidence append
        ↓
Graph Updates
```

---

## 3. Package Structure

```
bot-knows/
├── src/
│   ├── bot_knows/
│   │   ├── __init__.py
│   │   ├── config.py                 # Pydantic BaseSettings
│   │   ├── logging.py
│   │   ├── domain/
│   │   │   ├── chat.py               # Internal Chat entity
│   │   │   ├── message.py            # User–Assistant Message
│   │   │   ├── topic.py              # Canonical Topic
│   │   │   ├── topic_evidence.py     # Append-only evidence
│   │   │   └── relation.py
│   │   ├── models/                   # Public immutable DTOs
│   │   │   ├── chat.py
│   │   │   ├── message.py
│   │   │   ├── topic.py
│   │   │   └── recall.py
│   │   ├── interfaces/
│   │   ├── services/
│   │   │   ├── chat_processing.py    # Chat creation + classification
│   │   │   ├── message_builder.py
│   │   │   ├── topic_extraction.py
│   │   │   ├── dedup_service.py
│   │   │   ├── graph_service.py
│   │   │   └── tasks.py
│   │   ├── infra/
│   │   ├── importers/
│   │   │   ├── base.py               # ChatImportAdapter ABC
│   │   │   ├── registry.py
│   │   │   ├── chatgpt.py
│   │   │   ├── claude.py
│   │   │   └── generic_json.py
│   │   └── utils/
│   │       └── hashing.py
├── tests/
├── pyproject.toml
└── README.md
```

---

## 4. Ingestion Boundary Model (MOST IMPORTANT)

There is **no canonical ChatIngest model**.

Importers only normalize provider data into a **frozen ingestion object**.
They do **not**:
- create Chats or Messages
- generate IDs
- classify
- persist data

### ChatIngest (Importer Output)

```python
class ChatIngest(BaseModel, frozen=True):
    source: str
    imported_chat_timestamp: int  # epoch seconds
    title: str | None
    messages: list[IngestMessage]
```

```python
class IngestMessage(BaseModel, frozen=True):
    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: int  # epoch seconds
    chat_id: str
```

---

## 5. Import Adapters

### Purpose

Import adapters:
- isolate provider-specific parsing
- protect the domain model
- allow community extensions

### Base Import Adapter Interface

```python
class ChatImportAdapter(ABC):
    @abstractmethod
    def parse(self, raw_export: dict) -> ChatIngest:
        """
        Convert provider-specific export into ChatIngest objects.
        Must not persist or mutate state.
        """
```

---

## 6. Chat Creation & Classification (Domain)

### Chat Identity

```
chat_id = sha256(title + source + imported_chat_timestamp)
```

### Persistent Chat Model

```python
class Chat(BaseModel):
    id: str
    title: str
    source: str
    category: ChatCategory
    tags: list[str]
    created_on: int
```

### Title Resolution
- Use imported title if present
- Otherwise: first sentence of the first message

### Classification
- Executed **once**, only if Chat does not exist
- Input:
  - First user–assistant pair
  - Last user–assistant pair
- Output:
  - Single-label `ChatCategory`
  - Free-form tags

---

## 7. Message Creation & Ordering

Messages are derived from ordered ingest messages and stored separately.

```python
class Message(BaseModel):
    message_id: str
    chat_id: str
    user_content: str = ""
    assistant_content: str = ""
    created_on: int
```

Rules:
- Missing side defaults to empty string
- Hash-based `message_id`
- Ordering stored only via graph edge `FOLLOWS_AFTER`

---

## 8. Topic System (Semantic Concepts)

Topics are canonical, reusable semantic units extracted from Messages.

### Topic (Canonical)

```python
class Topic(BaseModel):
    topic_id: str
    canonical_name: str
    embedding: list[float]
    importance: float
    recall_strength: float
```

### TopicEvidence (Append-only)

```python
class TopicEvidence(BaseModel):
    evidence_id: str
    topic_id: str
    extracted_name: str
    source_message_id: str
    confidence: float
    timestamp: int
```

Evidence is **never merged or deleted**.

---

## 9. Semantic Deduplication

1. Extract candidate Topic from Message
2. Compare embedding against existing Topics

Thresholds:
- `>= 0.92` → same Topic
- `0.80–0.92` → new Topic + `POTENTIALLY_DUPLICATE_OF`
- `< 0.80` → new Topic

Importance and recall are incremented only.

---

## 10. Graph Model

### Nodes
- Chat
- Message
- Topic

### Edges

```
(Message)-[:IS_PART_OF]->(Chat)
(Message)-[:FOLLOWS_AFTER]->(Message)
(Topic)-[:IS_SUPPORTED_BY]->(Message)
```

Optional:
```
(Topic)-[:POTENTIALLY_DUPLICATE_OF]->(Topic)
```

---

## 11. Interfaces & Infra

Interfaces define contracts for embeddings, graph, recall, and storage.

Infra provides concrete implementations for MongoDB, Neo4j, Redis, and LLM providers.

---

## 12. Configuration

Uses **pydantic-settings** with split configuration objects.

All timestamps use **epoch seconds (int)**.

Python **>= 3.13**.

---

## 13. Design Rules (Non‑Negotiable)

1. Importers never persist
2. No canonical event model
3. Chats contain no message content
4. Messages contain all content
5. Topics are canonical; evidence is append-only
6. Graph defines ordering and semantics
7. Use new version of dependecy packages

---

## 14. Why this design is future‑proof

- New chat platforms → new adapter
- Better AI models → reprocess Messages
- New topic logic → no data loss
- UI/API changes → no core changes

---

## 15. Exposed/public

1. bot_knows/__init__.py

Expose only high-level entry points and config.

__all__ = [
    "config",
    "services",
    "importers",
    "models",
    "interfaces",
]


Do not expose domain, infra, or utils.

2. Importers (Public Extension API)
bot_knows/importers/__init__.py
__all__ = [
    "ChatImportAdapter",
    "ImportAdapterRegistry",
]

bot_knows/importers/base.py
__all__ = [
    "ChatImportAdapter",
]

bot_knows/importers/registry.py
__all__ = [
    "ImportAdapterRegistry",
]


Concrete adapters (chatgpt.py, claude.py)
→ NO __all__ (examples, not contracts)

3. Ingestion Models (Public Boundary)
bot_knows/models/ingest.py
__all__ = [
    "ChatIngest",
    "IngestMessage",
]


4. Public DTO Models
bot_knows/models/chat.py
__all__ = [
    "Chat",
    "ChatCategory",
]

bot_knows/models/message.py
__all__ = [
    "Message",
]

bot_knows/models/topic.py
__all__ = [
    "Topic",
    "TopicEvidence",
]

bot_knows/models/recall.py
__all__ = [
    "RecallItem",
]

5. Interfaces (DI Contracts)
bot_knows/interfaces/__init__.py
__all__ = [
    "EmbeddingServiceInterface",
    "GraphServiceInterface",
    "RecallServiceInterface",
    "StorageInterface",
    "LLMInterface",
]


Each interface file should mirror this.

Example (embedding.py):

__all__ = [
    "EmbeddingServiceInterface",
]

6. Services (Public Entry Points Only)
bot_knows/services/__init__.py
__all__ = [
    "ChatProcessingService",
    "RecallService",
]

7. Config (Public)
bot_knows/config.py
__all__ = [
    "MongoSettings",
    "LLMSettings",
    "GraphSettings",
    "RedisSettings",
]

## Final Sanity Check for Exposed/Public

If a user can do this:

from bot_knows import *

They should see only:

- config

- importers

- models

- interfaces

- top-level services

## 15. Dependency manager 
- uv
- packaging: 
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"