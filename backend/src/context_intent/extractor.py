"""Intent extraction from conversational input."""

import logging
from typing import List, Dict, Optional

from .schema import DeploymentIntent, ConversationMessage
from ..llm.ollama_client import OllamaClient
from ..llm.prompts import build_intent_extraction_prompt, INTENT_EXTRACTION_SCHEMA

logger = logging.getLogger(__name__)


class IntentExtractor:
    """Extract structured deployment intent from natural language conversation."""

    def __init__(self, llm_client: Optional[OllamaClient] = None):
        """
        Initialize intent extractor.

        Args:
            llm_client: Optional Ollama client (creates default if not provided)
        """
        self.llm_client = llm_client or OllamaClient()

    def extract_intent(
        self,
        user_message: str,
        conversation_history: Optional[List[ConversationMessage]] = None
    ) -> DeploymentIntent:
        """
        Extract deployment intent from user message.

        Args:
            user_message: Latest user message
            conversation_history: Optional previous conversation context

        Returns:
            DeploymentIntent with extracted requirements

        Raises:
            ValueError: If extraction fails or LLM returns invalid data
        """
        # Convert conversation history to dict format for LLM
        history_dicts = None
        if conversation_history:
            history_dicts = [
                {"role": msg.role, "content": msg.content}
                for msg in conversation_history
            ]

        # Build extraction prompt
        prompt = build_intent_extraction_prompt(user_message, history_dicts)

        try:
            # Extract structured data from LLM
            extracted = self.llm_client.extract_structured_data(
                prompt,
                INTENT_EXTRACTION_SCHEMA,
                temperature=0.3  # Lower temperature for more consistent extraction
            )

            # Validate and parse into Pydantic model
            intent = self._parse_extracted_intent(extracted)
            logger.info(f"Extracted intent: use_case={intent.use_case}, users={intent.user_count}")

            return intent

        except Exception as e:
            logger.error(f"Failed to extract intent: {e}")
            raise ValueError(f"Intent extraction failed: {e}")

    def _parse_extracted_intent(self, raw_data: Dict) -> DeploymentIntent:
        """
        Parse and validate raw LLM output into DeploymentIntent.

        Args:
            raw_data: Raw dict from LLM

        Returns:
            Validated DeploymentIntent

        Raises:
            ValueError: If data is invalid
        """
        # Handle common LLM mistakes
        cleaned_data = self._clean_llm_output(raw_data)

        try:
            return DeploymentIntent(**cleaned_data)
        except Exception as e:
            logger.error(f"Failed to parse intent from: {cleaned_data}")
            raise ValueError(f"Invalid intent data: {e}")

    def _clean_llm_output(self, data: Dict) -> Dict:
        """
        Clean common LLM output mistakes.

        Args:
            data: Raw LLM output

        Returns:
            Cleaned data dict
        """
        cleaned = data.copy()

        # Fix use_case if it contains the full enum string
        if "use_case" in cleaned and "|" in str(cleaned["use_case"]):
            # LLM sometimes returns "chatbot|customer_service|..." instead of just "chatbot"
            # Take the first option
            cleaned["use_case"] = cleaned["use_case"].split("|")[0].strip()

        # Ensure domain_specialization is a list
        if "domain_specialization" in cleaned:
            if isinstance(cleaned["domain_specialization"], str):
                # Convert single string to list
                cleaned["domain_specialization"] = [cleaned["domain_specialization"]]
            elif "|" in str(cleaned.get("domain_specialization", "")):
                # Handle "general|code" format
                cleaned["domain_specialization"] = [
                    d.strip() for d in cleaned["domain_specialization"].split("|")
                ]

        # Remove any unexpected fields that aren't in the schema
        valid_fields = DeploymentIntent.model_fields.keys()
        cleaned = {k: v for k, v in cleaned.items() if k in valid_fields}

        return cleaned

    def infer_missing_fields(self, intent: DeploymentIntent) -> DeploymentIntent:
        """
        Infer missing optional fields based on available information.

        Args:
            intent: Partially filled intent

        Returns:
            Intent with inferred fields
        """
        # If no throughput priority specified, infer from use case
        throughput_map = {
            "batch_analytics": "very_high",
            "summarization": "high",
            "qa_retrieval": "high",
            "code_generation": "medium",
            "chatbot": "medium",
            "customer_service": "medium",
            "content_creation": "low"
        }

        if intent.throughput_priority == "medium":  # default value
            inferred = throughput_map.get(intent.use_case, "medium")
            intent.throughput_priority = inferred  # type: ignore

        # Infer domain specialization from use case if not specified
        if intent.domain_specialization == ["general"]:
            if intent.use_case == "code_generation":
                intent.domain_specialization = ["general", "code"]
            elif "multilingual" in intent.additional_context.lower() if intent.additional_context else False:
                intent.domain_specialization = ["general", "multilingual"]

        return intent
