"""Prompt templates for LLM interactions."""


INTENT_EXTRACTION_SCHEMA = """
Expected JSON schema:
{
  "use_case": "chatbot|customer_service|summarization|code_generation|content_creation|qa_retrieval|batch_analytics",
  "user_count": <integer>,
  "latency_requirement": "very_high|high|medium|low",
  "throughput_priority": "very_high|high|medium|low",
  "budget_constraint": "strict|moderate|flexible|none",
  "domain_specialization": ["general"|"code"|"multilingual"|"enterprise"],
  "additional_context": "<any other relevant details mentioned>"
}
"""


def build_intent_extraction_prompt(user_message: str, conversation_history: list = None) -> str:
    """
    Build prompt for extracting deployment intent from user conversation.

    Args:
        user_message: Latest user message
        conversation_history: Optional list of previous messages

    Returns:
        Formatted prompt string
    """
    context = ""
    if conversation_history:
        context = "Previous conversation:\n"
        for msg in conversation_history[-3:]:  # Last 3 messages for context
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            context += f"{role}: {content}\n"
        context += "\n"

    prompt = f"""You are an expert AI assistant helping users deploy Large Language Models (LLMs) for production use cases.

{context}Current user message: {user_message}

Your task is to extract structured information about their deployment requirements. Analyze what they've said and infer:

1. **Use case**: What type of application (chatbot, customer service, code generation, summarization, etc.)
2. **User count**: How many users or scale mentioned (estimate if not explicit)
3. **Latency requirement**: How important is low latency? (very_high = sub-500ms, high = sub-2s, medium = 2-5s, low = >5s acceptable)
4. **Throughput priority**: Is high request volume more important than low latency?
5. **Budget constraint**: How price-sensitive are they?
6. **Domain specialization**: Any specific domains mentioned (code, multilingual, enterprise, etc.)

Be intelligent about inference:
- "thousands of users" → estimate specific number
- "needs to be fast" or "low latency critical" → latency_requirement: very_high
- "can tolerate higher latency" or "quality over speed" → latency_requirement: medium or low
- "batch processing" → throughput_priority: very_high, latency_requirement: low
- "customer-facing" → latency_requirement: high or very_high
- "budget is flexible" or "no budget constraint" → budget_constraint: flexible or none
- No budget mentioned → budget_constraint: moderate
- "cost-sensitive" or "cost efficiency important" → budget_constraint: strict or moderate

{INTENT_EXTRACTION_SCHEMA}
"""
    return prompt


CONVERSATIONAL_RESPONSE_TEMPLATE = """You are a helpful AI assistant for the AI Pre-Deployment Assistant.

The user is working on deploying a Large Language Model for their use case. You are here to have a natural conversation with them to understand their needs.

Current context:
{context}

User message: {user_message}

Based on what we know so far:
{current_understanding}

Respond naturally to the user. If we still need critical information (use case, scale, latency requirements), ask clarifying questions in a conversational way. If we have enough information, let them know we're ready to generate recommendations.

Keep your response concise (2-3 sentences max).
"""


def build_conversational_prompt(
    user_message: str,
    current_understanding: dict,
    conversation_history: list = None
) -> str:
    """
    Build prompt for conversational AI responses.

    Args:
        user_message: Latest user message
        current_understanding: Current extracted deployment intent
        conversation_history: Previous conversation messages

    Returns:
        Formatted prompt
    """
    context = ""
    if conversation_history:
        context = "Previous messages:\n"
        for msg in conversation_history[-2:]:
            role = msg.get('role', 'user')
            content = msg.get('content', '')
            context += f"- {role}: {content}\n"

    understanding = ""
    if current_understanding:
        understanding = f"""- Use case: {current_understanding.get('use_case', 'unknown')}
- User count: {current_understanding.get('user_count', 'unknown')}
- Latency requirement: {current_understanding.get('latency_requirement', 'unknown')}
"""

    return CONVERSATIONAL_RESPONSE_TEMPLATE.format(
        context=context,
        user_message=user_message,
        current_understanding=understanding
    )


YAML_EXPLANATION_TEMPLATE = """Explain the following KServe deployment configuration in simple terms for a user who may not be familiar with Kubernetes:

{yaml_content}

Provide a brief 2-3 sentence explanation of:
1. What model is being deployed
2. What GPU resources are being used
3. Key configuration settings (replicas, scaling, etc.)

Keep it non-technical and focused on the business value.
"""
