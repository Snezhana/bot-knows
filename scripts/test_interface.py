import asyncio
import sys
from pathlib import Path

from bot_knows import (
    BotKnows,
    ClaudeAdapter,
    MongoStorageRepository,
    Neo4jGraphRepository,
    OpenAIProvider,
)

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Simple usage - config loaded from .env automatically
async def main() -> None:
    async with BotKnows(
        storage_class=MongoStorageRepository,
        graphdb_class=Neo4jGraphRepository,
        llm_class=OpenAIProvider,
    ) as bk:
        # fixture_path = Path(__file__).parent.parent / "tests" / "fixtures" / "claude_export.json"

        # result = await bk.insert_chats(fixture_path, ClaudeAdapter)
        topics = await bk.get_topic_evidence(
            "48a79885e74239cedf4fa97053d56034b35e56957ead7a9c100271a84854157e"
        )
        print(topics)


if __name__ == "__main__":
    asyncio.run(main())
