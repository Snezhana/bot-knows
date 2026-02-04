"""Chat processing service for bot_knows.

This module provides the main service for chat creation and classification.
"""

from bot_knows.interfaces.llm import LLMInterface
from bot_knows.interfaces.storage import StorageInterface
from bot_knows.logging import get_logger
from bot_knows.models.chat import ChatCategory, ChatDTO
from bot_knows.models.ingest import ChatIngest, IngestMessage
from bot_knows.utils.hashing import generate_chat_id

__all__ = [
    "ChatProcessingService",
]

logger = get_logger(__name__)


class ChatProcessingService:
    """Service for chat creation and classification.

    Processes ChatIngest objects into ChatDTO objects, handling:
    - Chat identity resolution (deterministic ID generation)
    - Title resolution (from import or first message)
    - One-time classification (only for new chats)

    Example:
        service = ChatProcessingService(storage, llm)
        chat, is_new = await service.process(chat_ingest)
    """

    def __init__(
        self,
        storage: StorageInterface,
        llm: LLMInterface,
    ):
        """Initialize service with dependencies.

        Args:
            storage: Storage interface for persistence
            llm: LLM interface for classification
        """
        self._storage = storage
        self._llm = llm

    async def process(self, chat_ingest: ChatIngest) -> tuple[ChatDTO, bool]:
        """Process chat ingest into ChatDTO.

        If chat already exists (by ID), returns existing chat.
        Classification only runs for new chats.

        Args:
            chat_ingest: Ingested chat data

        Returns:
            Tuple of (ChatDTO, is_new) where is_new indicates
            if this was a newly created chat
        """
        # Resolve title first (needed for ID)
        title = self._resolve_title(chat_ingest)

        # Generate deterministic chat ID
        chat_id = generate_chat_id(
            title=title,
            source=chat_ingest.source,
            timestamp=chat_ingest.imported_chat_timestamp,
        )

        # Check if already exists (idempotency)
        existing = await self._storage.get_chat(chat_id)
        if existing:
            logger.debug("chat_already_exists", chat_id=chat_id)
            return existing, False

        # Classify new chat
        category, tags = await self._classify(chat_ingest)

        # Create ChatDTO
        chat = ChatDTO(
            id=chat_id,
            title=title,
            source=chat_ingest.source,
            category=category,
            tags=tags,
            created_on=chat_ingest.imported_chat_timestamp,
        )

        # Persist
        await self._storage.save_chat(chat)

        logger.info(
            "chat_created",
            chat_id=chat_id,
            title=title[:50],
            category=category.value,
            message_count=len(chat_ingest.messages),
        )

        return chat, True

    def _resolve_title(self, chat_ingest: ChatIngest) -> str:
        """Resolve chat title from ingest or first message.

        Priority:
        1. Title from import
        2. First sentence of first message
        3. "Untitled Chat" fallback
        """
        if chat_ingest.title:
            return chat_ingest.title

        # Use first sentence of first message
        for msg in chat_ingest.messages:
            if msg.content:
                # Extract first sentence (up to first period or 100 chars)
                content = msg.content.strip()
                period_idx = content.find(".")
                if period_idx > 0:
                    first_sentence = content[:period_idx]
                else:
                    first_sentence = content[:100]
                if first_sentence:
                    return first_sentence.strip()

        return "Untitled Chat"

    async def _classify(
        self,
        chat_ingest: ChatIngest,
    ) -> tuple[ChatCategory, list[str]]:
        """Classify chat using LLM.

        Uses first and last user-assistant pairs for classification.
        """
        messages = chat_ingest.messages

        # Find first user-assistant pair
        first_pair = self._find_pair(messages, from_start=True)

        # Find last user-assistant pair
        last_pair = self._find_pair(messages, from_start=False)

        if not first_pair:
            return ChatCategory.GENERAL, []

        # Use first pair for both if no distinct last pair
        if not last_pair or last_pair == first_pair:
            last_pair = first_pair

        try:
            return await self._llm.classify_chat(first_pair, last_pair)
        except Exception as e:
            logger.warning("classification_failed", error=str(e))
            return ChatCategory.GENERAL, []

    def _find_pair(
        self,
        messages: list[IngestMessage],
        from_start: bool,
    ) -> tuple[str, str] | None:
        """Find user-assistant pair from start or end of messages.

        Args:
            messages: List of ingest messages
            from_start: If True, search from start; else from end

        Returns:
            (user_content, assistant_content) tuple or None
        """
        if not messages:
            return None

        msg_iter = messages if from_start else reversed(messages)
        user_content: str | None = None

        for msg in msg_iter:
            if msg.role == "user" and user_content is None:
                user_content = msg.content
            elif msg.role == "assistant" and user_content is not None:
                return (user_content, msg.content)

        return None
