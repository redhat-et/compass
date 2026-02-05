"""Intent Extraction Service facade.

Provides a high-level interface for intent extraction operations.
"""

import logging

from ..llm.ollama_client import OllamaClient
from ..shared.schemas import ConversationMessage, DeploymentIntent
from .extractor import IntentExtractor

logger = logging.getLogger(__name__)


class IntentExtractionService:
    """High-level service for extracting deployment intent from natural language."""

    def __init__(self, llm_client: OllamaClient | None = None):
        """
        Initialize the Intent Extraction Service.

        Args:
            llm_client: Optional Ollama client for LLM operations
        """
        self.extractor = IntentExtractor(llm_client)

    def extract_and_infer(
        self,
        user_message: str,
        conversation_history: list[ConversationMessage] | None = None,
    ) -> DeploymentIntent:
        """
        Extract intent from user message and infer any missing fields.

        This is the primary method for obtaining a complete DeploymentIntent
        from natural language input.

        Args:
            user_message: The user's natural language description
            conversation_history: Optional previous conversation context

        Returns:
            Complete DeploymentIntent with extracted and inferred fields

        Raises:
            ValueError: If intent extraction fails
        """
        # Extract intent from natural language
        intent = self.extractor.extract_intent(user_message, conversation_history)

        # Infer any missing fields based on use case
        intent = self.extractor.infer_missing_fields(intent)

        logger.info(
            f"Intent extraction complete: use_case={intent.use_case}, "
            f"user_count={intent.user_count}, latency_priority={intent.latency_priority}"
        )

        return intent

    def extract_intent(
        self,
        user_message: str,
        conversation_history: list[ConversationMessage] | None = None,
    ) -> DeploymentIntent:
        """
        Extract intent from user message without inference.

        Use this when you need the raw extracted intent before inference.

        Args:
            user_message: The user's natural language description
            conversation_history: Optional previous conversation context

        Returns:
            DeploymentIntent with extracted fields only

        Raises:
            ValueError: If intent extraction fails
        """
        return self.extractor.extract_intent(user_message, conversation_history)

    def infer_missing_fields(self, intent: DeploymentIntent) -> DeploymentIntent:
        """
        Infer missing optional fields based on available information.

        Args:
            intent: Partially filled intent

        Returns:
            Intent with inferred fields
        """
        return self.extractor.infer_missing_fields(intent)
