"""Message builder service for bot_knows.

This module provides the service for building MessageDTOs from IngestMessages.
"""

from bot_knows.logging import get_logger
from bot_knows.models.ingest import IngestMessage
from bot_knows.models.message import MessageDTO
from bot_knows.utils.hashing import generate_message_id

__all__ = [
    "MessageBuilder",
]

logger = get_logger(__name__)


class MessageBuilder:
    """Service for building MessageDTOs from IngestMessages.

    Transforms a list of IngestMessages into MessageDTOs by:
    - Pairing user and assistant messages
    - Generating deterministic message IDs
    - Handling edge cases (missing pairs, system messages)

    Example:
        builder = MessageBuilder()
        messages = builder.build(ingest_messages, chat_id)
    """

    def build(
        self,
        ingest_messages: list[IngestMessage],
        chat_id: str,
    ) -> list[MessageDTO]:
        """Build MessageDTOs from IngestMessages.

        Pairs consecutive user-assistant messages into single MessageDTO objects.
        System messages are stored with empty user_content.

        Args:
            ingest_messages: List of ingested messages
            chat_id: Parent chat ID

        Returns:
            List of MessageDTO objects
        """
        if not ingest_messages:
            return []

        # Sort by timestamp to ensure correct ordering
        sorted_messages = sorted(ingest_messages, key=lambda m: m.timestamp)

        messages: list[MessageDTO] = []
        pending_user: IngestMessage | None = None

        for msg in sorted_messages:
            if msg.role == "system":
                # System messages become standalone with empty user_content
                message_dto = self._create_message(
                    chat_id=chat_id,
                    user_content="",
                    assistant_content=msg.content,
                    timestamp=msg.timestamp,
                )
                messages.append(message_dto)

            elif msg.role == "user":
                # If we have a pending user message, create it as standalone
                if pending_user:
                    message_dto = self._create_message(
                        chat_id=chat_id,
                        user_content=pending_user.content,
                        assistant_content="",
                        timestamp=pending_user.timestamp,
                    )
                    messages.append(message_dto)

                pending_user = msg

            elif msg.role == "assistant":
                # Pair with pending user message if available
                user_content = pending_user.content if pending_user else ""
                timestamp = pending_user.timestamp if pending_user else msg.timestamp

                message_dto = self._create_message(
                    chat_id=chat_id,
                    user_content=user_content,
                    assistant_content=msg.content,
                    timestamp=timestamp,
                )
                messages.append(message_dto)
                pending_user = None

        # Handle trailing user message
        if pending_user:
            message_dto = self._create_message(
                chat_id=chat_id,
                user_content=pending_user.content,
                assistant_content="",
                timestamp=pending_user.timestamp,
            )
            messages.append(message_dto)

        logger.debug(
            "messages_built",
            chat_id=chat_id,
            input_count=len(ingest_messages),
            output_count=len(messages),
        )

        return messages

    def _create_message(
        self,
        chat_id: str,
        user_content: str,
        assistant_content: str,
        timestamp: int,
    ) -> MessageDTO:
        """Create a MessageDTO with deterministic ID."""
        message_id = generate_message_id(
            chat_id=chat_id,
            user_content=user_content,
            assistant_content=assistant_content,
            timestamp=timestamp,
        )

        return MessageDTO(
            message_id=message_id,
            chat_id=chat_id,
            user_content=user_content,
            assistant_content=assistant_content,
            created_on=timestamp,
        )
