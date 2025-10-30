"""Data schemas for deployment intent and specifications."""

from typing import Literal

from pydantic import BaseModel, Field


class TrafficProfile(BaseModel):
    """GuideLLM traffic profile for the deployment."""

    prompt_tokens: int = Field(..., description="Target prompt length in tokens (GuideLLM config)")
    output_tokens: int = Field(..., description="Target output length in tokens (GuideLLM config)")
    expected_qps: float | None = Field(None, description="Expected queries per second")


class SLOTargets(BaseModel):
    """Service Level Objective targets for the deployment (p95 percentiles)."""

    ttft_p95_target_ms: int = Field(..., description="Time to First Token p95 target (ms)")
    itl_p95_target_ms: int = Field(..., description="Inter-Token Latency p95 target (ms/token)")
    e2e_p95_target_ms: int = Field(..., description="End-to-end latency p95 target (ms)")


class GPUConfig(BaseModel):
    """GPU configuration specification."""

    gpu_type: str = Field(..., description="GPU type (e.g., NVIDIA-L4, NVIDIA-A100-80GB)")
    gpu_count: int = Field(..., description="Total number of GPUs")
    tensor_parallel: int = Field(1, description="Tensor parallelism degree")
    replicas: int = Field(1, description="Number of independent replicas")


class DeploymentIntent(BaseModel):
    """Extracted deployment requirements from user conversation."""

    use_case: Literal[
        "chatbot_conversational",
        "code_completion",
        "code_generation_detailed",
        "translation",
        "content_creation",
        "summarization_short",
        "document_analysis_rag",
        "long_document_summarization",
        "research_legal_analysis",
    ] = Field(..., description="Primary use case type")

    experience_class: Literal["instant", "conversational", "interactive", "deferred", "batch"] = Field(
        ..., description="User experience class defining latency expectations"
    )

    user_count: int = Field(..., description="Number of users or scale")

    latency_requirement: Literal["very_high", "high", "medium", "low"] = Field(
        ...,
        description="Latency sensitivity (very_high=sub-500ms, high=sub-2s, medium=2-5s, low=>5s)",
    )

    throughput_priority: Literal["very_high", "high", "medium", "low"] = Field(
        default="medium", description="Importance of high request volume"
    )

    budget_constraint: Literal["strict", "moderate", "flexible", "none"] = Field(
        default="moderate", description="Cost sensitivity"
    )

    domain_specialization: list[str] = Field(
        default_factory=lambda: ["general"],
        description="Domain requirements (general, code, multilingual, enterprise)",
    )

    additional_context: str | None = Field(
        None, description="Any other relevant details from conversation"
    )


class DeploymentRecommendation(BaseModel):
    """Complete deployment recommendation with all specifications."""

    model_config = {"protected_namespaces": ()}

    # Input intent
    intent: DeploymentIntent

    # Generated specifications
    traffic_profile: TrafficProfile
    slo_targets: SLOTargets

    # Recommended configuration
    model_id: str = Field(..., description="Recommended model identifier")
    model_name: str = Field(..., description="Human-readable model name")
    gpu_config: GPUConfig

    # Performance predictions
    predicted_ttft_p95_ms: int
    predicted_itl_p95_ms: int
    predicted_e2e_p95_ms: int
    predicted_throughput_qps: float

    # Cost estimation
    cost_per_hour_usd: float
    cost_per_month_usd: float

    # Metadata
    meets_slo: bool = Field(..., description="Whether configuration meets SLO targets")
    reasoning: str = Field(..., description="Explanation of recommendation choice")
    alternative_options: list[dict] | None = Field(
        default=None, description="Alternative configurations with trade-offs"
    )


class ConversationMessage(BaseModel):
    """Single message in the conversation history."""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: str | None = None
