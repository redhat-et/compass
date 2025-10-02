# System-Level Architecture for Pre-Deployment Assistant

## Overview

This document defines the system architecture for the **Pre-Deployment Assistant for LLM Serving** capability, which streamlines the path from concept to production-ready endpoint for Red Hat AI (RHAI) customers.

## Project Context

**Vision**: A world where every organization adopts AI at scale, with a personalized AI stack optimized for its business needs.

**Mission**: Ensuring our customers fulfill their AI goals and maximize ROI of their GenAI investments through deep visibility and personalized recommendations.

**Problem Statement**: Deploying LLMs in production environments remains complex and error-prone:
- Application Developers struggle to translate business needs into infrastructure choices
- Cluster Admins face overprovisioning risks without guidance on aligning workloads to budgets and SLAs
- Trial-and-error is the default, extending time-to-production and inflating costs

## Solution Approach

The assistant follows a **4-stage conversational flow**:

1. **Understand Business Context** - Capture task definition, expected load, and constraints
2. **Provide Tailored Recommendations** - Suggest models, hardware, and SLOs
3. **Enable Interactive Exploration** - Support what-if scenario analysis
4. **One-Click Deployment** - Generate and deploy production-ready configurations

---

## Architecture Components

### 1. Conversational Interface Layer
**Purpose**: Natural language dialogue to capture user intent

**Technology Options**:
- **Chainlit** - Python-first conversational UI framework, good for prototyping
- **Streamlit + LangChain** - Rapid development with existing LLM orchestration
- **Custom React + WebSocket backend** - More flexible, production-grade UI control

**Responsibilities**:
- Render conversational UI
- Handle user input/output
- Maintain session state
- Display recommendations and simulation results

---

### 2. Context & Intent Engine
**Purpose**: Extract structured intent from conversational input and generate comprehensive deployment specifications

**Technology Options**:
- **LangChain with structured output** - Built-in prompt engineering for extraction
- **Guardrails AI** - Schema validation and structured extraction from LLMs
- **Custom LLM + Pydantic validators** - Full control with type-safe extraction

**Key Functions**:
- Parse user inputs into structured data (task definition, expected load, priorities)
- Auto-generate traffic characteristics from high-level descriptions
- Map use cases to appropriate default SLOs
- Validate completeness before proceeding to recommendations
- Store conversation context for iterative refinement
- Present editable specification to user for review/modification

**Enhanced Data Schema**:
```python
class DeploymentIntent:
    # High-level intent (user-provided)
    task_type: str  # "chatbot", "summarization", "code-generation"
    use_case_description: str  # Natural language description
    subject_matter: Optional[str]  # May influence model selection

    # Workload characteristics (auto-inferred or user-specified)
    expected_qps: Optional[float]
    concurrent_users: Optional[int]

    # Traffic profile (auto-generated from use case, editable by user)
    prompt_length_avg: Optional[int]  # Phase 1: point estimates
    prompt_length_dist: Optional[Distribution]  # Phase 2: full distribution
    generation_length_avg: Optional[int]
    generation_length_dist: Optional[Distribution]
    concurrency_steady_state: Optional[int]
    concurrency_peak: Optional[int]
    burstiness_pattern: Optional[str]  # "steady", "moderate", "high", "diurnal"
    request_heterogeneity: Optional[dict]  # mix of short/long, streaming vs non-streaming

    # Constraints
    priority: str  # "accuracy", "latency", "cost", "balanced"
    budget: Optional[BudgetConstraint]

    # SLOs (auto-suggested from use case template, user-editable)
    target_ttft_p90_ms: Optional[float]  # Time to First Token
    target_tpot_p90_ms: Optional[float]  # Time Per Output Token
    target_e2e_latency_p95_ms: Optional[float]  # End-to-End latency
    target_throughput_rps: Optional[float]  # Requests per second
    quality_threshold: Optional[str]  # "standard", "high", "very_high"
    reliability_target: Optional[float]  # e.g., 99.9% uptime

    # Optional user preferences
    preferred_models: Optional[List[str]]  # User can specify models to consider
    available_gpus: Optional[List[str]]  # User can specify available GPU types
    gpu_constraints: Optional[dict]  # max count, tensor parallelism preferences

class BudgetConstraint:
    max_monthly_cost_usd: Optional[float]
    max_hourly_cost_usd: Optional[float]
    cost_per_1k_tokens: Optional[float]
```

**Use Case → SLO Mapping**:
The engine maintains templates for common use cases:
```python
USE_CASE_TEMPLATES = {
    "interactive_chatbot": {
        "default_slos": {
            "ttft_p90_ms": 300,
            "tpot_p90_ms": 50,
            "e2e_latency_p95_ms": 2000,
            "quality": "high"
        },
        "traffic_defaults": {
            "prompt_length_avg": 150,
            "generation_length_avg": 200,
            "burstiness_pattern": "moderate"
        }
    },
    "document_summarization": {
        "default_slos": {
            "ttft_p90_ms": 1000,
            "tpot_p90_ms": 100,
            "e2e_latency_p95_ms": 10000,
            "quality": "very_high"
        },
        "traffic_defaults": {
            "prompt_length_avg": 2000,
            "generation_length_avg": 500,
            "burstiness_pattern": "low"
        }
    },
    "code_generation": {
        "default_slos": {
            "ttft_p90_ms": 500,
            "tpot_p90_ms": 75,
            "e2e_latency_p95_ms": 5000,
            "quality": "high"
        },
        "traffic_defaults": {
            "prompt_length_avg": 300,
            "generation_length_avg": 400,
            "burstiness_pattern": "moderate"
        }
    }
}
```

---

### 3. Recommendation Engine
**Purpose**: Suggest model, hardware, and SLO configurations based on captured context and deployment specifications

**Technology Options**:
- **Rule-based decision tree + LLM augmentation** - Hybrid approach with explainability
- **Vector similarity search (FAISS/Pinecone) + retrieval** - Match context to known deployment patterns
- **Custom ML model** - Train on historical deployment data (future iteration)

**Sub-Components**:

#### 3a. Traffic Profile Generator
Translates high-level use case descriptions into concrete traffic characteristics.

**Inputs**:
- Use case type (chatbot, summarization, etc.)
- User count and concurrency estimates
- Business context

**Outputs**:
- Prompt/generation length estimates (Phase 1: averages, Phase 2: distributions)
- QPS estimates (steady-state and peak)
- Burstiness patterns
- Request heterogeneity profile

**Implementation Approach (Phase 1)**:
- Rule-based heuristics using use case templates
- LLM-assisted estimation for edge cases
- Simple point estimates (mean values only)

**Phase 2 Enhancements**:
- Full statistical distributions
- Industry benchmark lookup from Knowledge Base
- Pattern learning from historical deployments

#### 3b. Model Recommendation Engine
Filters and ranks models based on task compatibility and user constraints.

**Inputs**:
- Deployment specification (from Context Engine)
- Traffic profile
- SLO targets
- User preferences (optional preferred models, subject matter domain)

**Recommendation Logic**:
1. Filter models by task compatibility and subject matter domain
2. Consider user-specified model preferences (if any)
3. Apply curated model constraints (approved model catalog)
4. Evaluate performance vs accuracy trade-offs against SLOs
5. Query benchmark data for each candidate model
6. Rank by user priority (cost vs latency vs accuracy)
7. Generate explanation with rationale for each recommendation

**Outputs**:
- Ranked list of model recommendations with detailed rationale
- Performance projections for each model
- Trade-off analysis (accuracy vs cost vs latency)

#### 3c. Capacity Planning Engine
Determines GPU requirements (type, count, configuration) for each model recommendation.

**Inputs**:
- Selected model (or candidate model for evaluation)
- Traffic profile
- SLO targets
- Available GPU types (optional user constraint)
- Budget constraints

**Capacity Calculation Logic**:
```python
def calculate_capacity(
    model: Model,
    gpu_type: GPUType,
    traffic_profile: TrafficProfile,
    slo_targets: SLOTargets
) -> CapacityPlan:
    """
    Returns:
    - num_gpus_required: int
    - deployment_strategy: "independent" | "tensor_parallel"
    - tensor_parallel_degree: Optional[int]
    - expected_throughput_rps: float
    - predicted_slo_compliance: Dict[str, bool]  # TTFT, TPOT, E2E
    - estimated_cost: CostEstimate
    """
    # Phase 1: Benchmark data interpolation
    # 1. Query: (model, gpu_type) -> throughput_per_gpu, latency metrics
    # 2. Check if model fits on single GPU (memory constraint)
    # 3. If not, require tensor parallelism
    # 4. Calculate: num_gpus = ceil(traffic_profile.peak_qps / throughput_per_gpu)
    # 5. Validate SLO compliance using benchmark latency data
    # 6. Estimate cost: num_gpus * gpu_hourly_rate

    # Phase 2 enhancements:
    # - Optimize batch size for throughput/latency trade-off
    # - Consider auto-scaling (min/max GPU count)
    # - Advanced tensor parallelism optimization
```

**Benchmark Data Requirements**:
```json
{
  "model": "llama-3-70b",
  "gpu_type": "A100-80GB",
  "tensor_parallel_degree": 4,
  "batch_size": 32,
  "metrics": {
    "throughput_tokens_per_sec": 3500,
    "ttft_p50_ms": 95,
    "ttft_p90_ms": 120,
    "ttft_p99_ms": 180,
    "tpot_p50_ms": 12,
    "tpot_p90_ms": 15,
    "tpot_p99_ms": 22,
    "memory_usage_gb": 280
  }
}
```

**Outputs**:
- GPU configuration (type, count, parallelism strategy)
- Deployment topology (independent replicas vs tensor parallelism)
- Cost estimates (hourly, monthly, per-1k-tokens)
- SLO compliance predictions

**Overall Recommendation Engine Outputs**:
- Ranked recommendations combining model + GPU configuration
- Example: "Llama-3-8B on 2x L4 GPUs: $X/month, meets TTFT<200ms, TPOT<50ms"
- Detailed trade-off analysis for top 3-5 options

---

### 4. Simulation & Exploration Layer
**Purpose**: Enable what-if analysis with cost/latency/throughput predictions and interactive specification editing

**Technology Options**:
- **Custom analytical models** - Formula-based estimation (tokens/sec × cost/token)
- **Discrete event simulation (SimPy)** - More accurate workload modeling (Phase 2)
- **Historical performance extrapolation** - Use real benchmark data interpolation

**Key Functions**:

#### Specification Review & Editing
- Display auto-generated deployment specification in user-friendly format
- Allow inline editing of traffic characteristics (prompt length, QPS, etc.)
- Allow modification of SLO targets (TTFT, TPOT, E2E latency)
- Re-trigger recommendations after manual specification changes

**UI Example**:
```
┌─────────────────────────────────────────────┐
│ Deployment Specification                    │
├─────────────────────────────────────────────┤
│ Use Case: Interactive Chatbot               │
│ Subject Matter: Customer Support            │
│ Expected Users: 1000 concurrent             │
│ Priority: Low Latency                       │
│                                             │
│ Traffic Profile (Auto-Generated):           │
│   Avg Prompt Length: 150 tokens [Edit]      │
│   Avg Response Length: 250 tokens [Edit]    │
│   Steady-State QPS: 50 [Edit]               │
│   Peak QPS: 100 [Edit]                      │
│   Burstiness: Moderate [Edit]               │
│                                             │
│ SLO Targets (Suggested):                    │
│   TTFT (p90): 200ms [Edit]                  │
│   TPOT (p90): 50ms [Edit]                   │
│   E2E Latency (p95): 2000ms [Edit]          │
│   Throughput: 100 rps [Edit]                │
│   Quality: High [Edit]                      │
│   Reliability: 99.9% [Edit]                 │
│                                             │
│ Budget Constraints:                         │
│   Max Monthly Cost: $5000 [Edit]            │
│                                             │
│ [Regenerate Recommendations] [Deploy]       │
└─────────────────────────────────────────────┘
```

#### What-If Scenario Analysis
- Modify model selection and see impact on cost/latency
- Change GPU types and see capacity/cost implications
- Adjust SLO targets and see which configurations remain viable
- Compare scenarios side-by-side

**Scenario Comparison Example**:
```
┌──────────────────────────────────────────────────────────────────┐
│ Scenario A              │ Scenario B              │ Difference   │
├──────────────────────────────────────────────────────────────────┤
│ Llama-3-8B              │ Llama-3-70B             │              │
│ 2x NVIDIA L4            │ 4x NVIDIA A100          │              │
│ $800/month              │ $2400/month             │ +$1600/month │
│ TTFT p90: 180ms ✓       │ TTFT p90: 150ms ✓       │ -30ms        │
│ TPOT p90: 45ms ✓        │ TPOT p90: 35ms ✓        │ -10ms        │
│ Quality: High           │ Quality: Very High      │ Better       │
│ Throughput: 120 rps ✓   │ Throughput: 200 rps ✓   │ +80 rps      │
└──────────────────────────────────────────────────────────────────┘
```

**Simulation Outputs**:
- Estimated costs (hourly, monthly, per-1k-tokens)
- SLO compliance predictions (TTFT, TPOT, E2E latency percentiles)
- Expected throughput (requests/sec, tokens/sec)
- Resource utilization estimates (GPU memory, compute)
- Quality projections (based on model capabilities)

---

### 5. Deployment Automation Engine
**Purpose**: Generate production-ready configs and trigger deployment

**Technology Options**:
- **Jinja2 templating + Python** - Generate KServe/vLLM YAML from templates
- **Kubernetes Python client** - Direct K8s API interaction for deployment
- **ArgoCD/Flux GitOps** - Push generated configs to Git for automated sync
- **Go + native Kubernetes client** - For advanced use cases requiring tighter K8s integration (Phase 2+)

**Phase 1 Recommendation: Python**

For the initial 3-month iteration, we recommend Python for these reasons:
- **Stack consistency**: Entire system uses Python (LangChain, Chainlit, LLM interactions)
- **Rapid development**: Jinja2 templating is native and highly productive
- **Mature client library**: Kubernetes Python client is feature-complete and well-maintained
- **Team expertise**: Likely more Python experience on AI-focused teams
- **Sufficient performance**: Not building performance-critical controllers or watch loops

**Phase 2+ Consideration: Go**

Consider migrating to Go if requirements evolve to include:
- **Custom Kubernetes operators**: Building CRDs with reconciliation loops
- **Advanced watch patterns**: Complex event-driven K8s resource management
- **Performance optimization**: High-volume deployment operations
- **Native K8s ecosystem**: Tighter integration with controller-runtime and operator-sdk
- **Smaller container footprint**: No Python runtime overhead

**Outputs**:
- KServe InferenceService YAML
- vLLM runtime configuration
- Autoscaling policies (HPA/KEDA configs)
- Observability setup (Prometheus ServiceMonitors, Grafana dashboards)

**Integration Points**:
- KServe API (for model serving)
- OpenShift/Kubernetes cluster
- Model registry (HuggingFace Hub, custom registry)

**Deployment Flow**:
1. Validate configuration against cluster resources
2. Generate KServe InferenceService manifest
3. Generate vLLM runtime parameters
4. Create autoscaling policies
5. Deploy to cluster
6. Configure observability hooks
7. Return deployment ID and monitoring links

---

### 6. Knowledge Base & Benchmarking Data Store
**Purpose**: Store performance data, industry standards, deployment patterns, and use case templates

**Technology Options**:
- **PostgreSQL + pgvector** - Relational data + vector search for similarity matching
- **MongoDB + Atlas Search** - Flexible schema for varied benchmark data
- **Specialized vector DB (Qdrant/Weaviate)** - Optimized for embedding-based retrieval

**Data Collections**:

#### 6a. Model Benchmarks
Comprehensive performance metrics for model + GPU combinations.

**Schema Requirements**:
```json
{
  "model_id": "meta-llama/Llama-3-70b",
  "gpu_type": "NVIDIA-A100-80GB",
  "tensor_parallel_degree": 4,
  "batch_size": 32,
  "quantization": null,  // or "int8", "int4", "awq"
  "metrics": {
    "throughput_tokens_per_sec": 3500,
    "throughput_requests_per_sec": 45,
    "ttft_p50_ms": 95,
    "ttft_p90_ms": 120,
    "ttft_p99_ms": 180,
    "tpot_p50_ms": 12,
    "tpot_p90_ms": 15,
    "tpot_p99_ms": 22,
    "e2e_latency_p50_ms": 450,
    "e2e_latency_p95_ms": 680,
    "e2e_latency_p99_ms": 920,
    "memory_usage_gb": 280,
    "gpu_utilization_pct": 85
  },
  "test_conditions": {
    "prompt_length_avg": 512,
    "generation_length_avg": 256,
    "concurrency": 32
  },
  "timestamp": "2025-01-15T00:00:00Z"
}
```

#### 6b. Hardware Profiles
GPU specifications, availability, and pricing.

```json
{
  "gpu_type": "NVIDIA-A100-80GB",
  "vram_gb": 80,
  "tflops_fp16": 312,
  "memory_bandwidth_gbps": 2039,
  "manufacturer": "NVIDIA",
  "architecture": "Ampere",
  "available": true,
  "supported_parallelism": ["tensor", "pipeline"],
  "max_tensor_parallel": 8
}
```

#### 6c. Cost Data
Pricing information for different GPU types.

```json
{
  "gpu_type": "NVIDIA-A100-80GB",
  "cloud_provider": "AWS",
  "instance_type": "p4d.24xlarge",
  "hourly_rate_usd": 32.77,
  "monthly_rate_usd": 23914.00,
  "region": "us-east-1",
  "effective_date": "2025-01-01"
}
```

#### 6d. Use Case SLO Templates
Default SLO targets and traffic profiles for common use cases.

```json
{
  "use_case": "interactive_chatbot",
  "description": "Real-time conversational assistant for customer support",
  "default_slos": {
    "ttft_p90_ms": 300,
    "tpot_p90_ms": 50,
    "e2e_latency_p95_ms": 2000,
    "quality": "high",
    "reliability_pct": 99.9
  },
  "traffic_defaults": {
    "prompt_length_avg": 150,
    "prompt_length_p95": 300,
    "generation_length_avg": 200,
    "generation_length_p95": 500,
    "burstiness_pattern": "moderate",
    "typical_concurrency_ratio": 0.1  // 10% of users active simultaneously
  },
  "recommended_model_characteristics": {
    "max_latency_priority": true,
    "min_size": "7B",
    "max_size": "70B",
    "quantization_acceptable": true
  }
}
```

#### 6e. Model Catalog
Curated list of approved models with metadata.

```json
{
  "model_id": "meta-llama/Llama-3-8b-instruct",
  "model_family": "Llama-3",
  "size_parameters": "8B",
  "context_window": 8192,
  "supported_tasks": ["chatbot", "summarization", "qa", "code-generation"],
  "subject_matter_domains": ["general", "technical"],
  "quality_tier": "high",
  "license": "Meta Community License",
  "approved_for_production": true,
  "minimum_vram_gb": 16,
  "supports_quantization": true
}
```

#### 6f. Deployment Templates
Pre-configured deployment patterns for common scenarios.

```json
{
  "template_id": "chatbot-low-latency-medium-scale",
  "use_case": "interactive_chatbot",
  "model_id": "meta-llama/Llama-3-8b-instruct",
  "gpu_type": "NVIDIA-L4",
  "gpu_count": 2,
  "tensor_parallel_degree": 1,
  "kserve_config": {
    "predictor": {
      "minReplicas": 2,
      "maxReplicas": 4
    }
  },
  "vllm_config": {
    "max_model_len": 4096,
    "gpu_memory_utilization": 0.9
  },
  "autoscaling_config": {
    "metric": "concurrency",
    "target": 10
  }
}
```

#### 6g. Deployment Outcomes
Historical data from real deployments for continuous learning.

```json
{
  "deployment_id": "deploy-abc123",
  "template_id": "chatbot-low-latency-medium-scale",
  "model_id": "meta-llama/Llama-3-8b-instruct",
  "gpu_type": "NVIDIA-L4",
  "gpu_count": 2,
  "actual_metrics": {
    "ttft_p90_ms": 185,
    "tpot_p90_ms": 42,
    "throughput_avg_rps": 125,
    "cost_per_hour_usd": 4.20
  },
  "slo_compliance": {
    "ttft": true,
    "tpot": true,
    "e2e_latency": true,
    "availability": 99.95
  },
  "traffic_observed": {
    "prompt_length_avg": 165,
    "generation_length_avg": 220,
    "peak_qps": 95
  },
  "deployed_at": "2025-01-10T00:00:00Z",
  "observation_period_days": 7
}
```

---

### 7. LLM Backend
**Purpose**: Power conversational AI and reasoning

**Technology Options**:
- **InstructLab-served model** - Aligned with Red Hat ecosystem
- **OpenAI API** - Fast prototyping, higher cost
- **Self-hosted Llama 3 via vLLM** - Full control, lower marginal cost

**Responsibilities**:
- Process natural language inputs
- Generate recommendations with reasoning
- Provide explanations for suggestions
- Handle multi-turn conversations

---

### 8. Orchestration & Workflow Engine
**Purpose**: Coordinate multi-step flows (conversation → recommendation → deployment)

**Technology Options**:
- **LangGraph** - State machine for LLM workflows
- **Temporal** - Robust workflow orchestration with retries/compensation
- **Custom FastAPI + state management** - Lightweight for initial iteration

**Workflow States**:
1. Context Collection
2. Recommendation Generation
3. Interactive Exploration
4. Configuration Finalization
5. Deployment Execution
6. Post-Deployment Validation

---

### 9. Inference Observability & SLO Monitoring
**Purpose**: Monitor deployed inference systems to validate SLO compliance and enable feedback loop to improve recommendations

**Technology Options**:
- **OpenTelemetry + Prometheus + Grafana** - Standard observability stack for metrics
- **RHAI Observability Platform** - Integrate with existing Red Hat AI observability
- **Custom metrics exporters** - vLLM/KServe-specific instrumentation

**Key Responsibilities**:

#### SLO Compliance Monitoring
Track whether deployed models meet their predicted SLO targets:
- **TTFT (Time to First Token)**: p50, p90, p99 latencies
- **TPOT (Time Per Output Token)**: p50, p90, p99 latencies
- **E2E Latency**: Full request-response time at p50, p95, p99
- **Throughput**: Actual requests/sec and tokens/sec vs predicted
- **Quality**: Model accuracy/quality metrics if available
- **Reliability**: Uptime, error rates, timeout rates

#### Resource Utilization Metrics
Monitor efficiency of GPU allocation:
- **GPU utilization**: Compute usage per GPU (target: >80% for cost efficiency)
- **GPU memory usage**: VRAM consumption vs available
- **Batch efficiency**: Average batch size achieved
- **Request queue depth**: Backlog indicating capacity constraints
- **Token throughput per GPU**: Measure of effective GPU usage

#### Cost Tracking
Validate cost predictions and identify optimization opportunities:
- **Actual cost per hour/month**: Compare to predicted
- **Cost per 1k tokens**: Unit economics
- **Cost per request**: Overhead analysis
- **Idle GPU time**: Identify overprovisioning

#### Traffic Pattern Analysis
Understand actual workload vs predictions:
- **Prompt length distribution**: Actual vs predicted (mean, p95, max)
- **Generation length distribution**: Actual vs predicted
- **QPS patterns**: Steady-state, peak, burstiness
- **Temporal patterns**: Diurnal cycles, weekday/weekend differences
- **Request heterogeneity**: Mix of short/long requests

**Integration Points**:
- **vLLM metrics endpoint**: Native Prometheus metrics for inference stats
- **KServe ServiceMonitor**: Kubernetes-native metrics collection
- **Grafana dashboards**: Pre-configured for SLO visualization
- **Alerting**: Proactive alerts when SLOs are violated or GPUs underutilized

**Feedback Loop to Knowledge Base**:
- Store actual performance metrics in Deployment Outcomes
- Compare predicted vs actual TTFT, TPOT, throughput, cost
- Identify systematic biases in predictions
- Update benchmark data and recommendation logic
- Enable continuous improvement of capacity planning accuracy

**Dashboard Example**:
```
┌─────────────────────────────────────────────────────────┐
│ Deployment: chatbot-prod-llama3-8b                      │
├─────────────────────────────────────────────────────────┤
│ SLO Compliance (Last 7 days):                           │
│   TTFT p90:    185ms ✓ (target: 200ms)                  │
│   TPOT p90:     48ms ✓ (target: 50ms)                   │
│   E2E p95:    1850ms ✓ (target: 2000ms)                 │
│   Throughput:  122 rps ✓ (target: 100 rps)              │
│   Uptime:     99.94% ✓ (target: 99.9%)                  │
│                                                         │
│ Resource Utilization:                                   │
│   GPU Usage:     78% ⚠️ (2x L4, could optimize)         │
│   GPU Memory:    14.2 / 24 GB per GPU                   │
│   Avg Batch:     18 requests                            │
│                                                         │
│ Cost Analysis:                                          │
│   Actual:       $812/month (predicted: $800)            │
│   Per 1k tok:   $0.042 (predicted: $0.040)              │
│                                                         │
│ Traffic Patterns:                                       │
│   Avg Prompt:   165 tokens (predicted: 150)             │
│   Avg Gen:      220 tokens (predicted: 200)             │
│   Peak QPS:     95 (predicted: 100)                     │
│                                                         │
│ Recommendation: GPU utilization below 80%.              │
│ Consider downsizing to 1x L4 to reduce cost by 50%.     │
└─────────────────────────────────────────────────────────┘
```

**Assistant Performance Metrics** (secondary focus):
- Recommendation acceptance rate
- Time-to-deployment
- Specification edit frequency (indicates auto-generation accuracy)

---

## Enhanced Data Flow Diagram

```
User: "I need a chatbot for 1000 users, low latency is critical"
    ↓
┌─────────────────────────────────────────────────────────────┐
│              Conversational Interface Layer                 │
└───────────────────────┬─────────────────────────────────────┘
                        │ Natural Language Input
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              Context & Intent Engine                        │
│  • Extract: task_type, users, priority                      │
│  • Lookup use case template → default SLOs                  │
│  • Generate traffic profile (prompt/gen length, QPS)        │
└───────────────────────┬─────────────────────────────────────┘
                        │ Deployment Specification
                        ↓
┌─────────────────────────────────────────────────────────────┐
│         Specification Review UI (Optional Edit)             │
│  • Display auto-generated spec                              │
│  • User can modify SLOs, traffic params                     │
└───────────────────────┬─────────────────────────────────────┘
                        │ Confirmed Specification
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              Recommendation Engine                          │
│  ┌──────────────────────────────────────────┐               │
│  │ Traffic Profile Generator                │               │
│  │  → QPS, prompt/gen lengths, burstiness   │               │
│  └──────────────────────────────────────────┘               │
│  ┌──────────────────────────────────────────┐               │
│  │ Model Recommendation Engine              │               │
│  │  → Filter by task, rank by priority      │               │
│  └──────────────────────────────────────────┘               │
│  ┌──────────────────────────────────────────┐               │
│  │ Capacity Planning Engine                 │               │
│  │  → Calculate GPU count, predict SLOs     │               │
│  └──────────────────────────────────────────┘               │
│              ↕ Query Knowledge Base                         │
│        (benchmarks, SLO templates, costs)                   │
└───────────────────────┬─────────────────────────────────────┘
                        │ Ranked Recommendations
                        ↓
         Present recommendations to user:
         "Llama-3-8B on 2x L4: $800/mo, TTFT<200ms"
         "Llama-3-70B on 4x A100: $2400/mo, better quality"
                        │
                        ↓
┌─────────────────────────────────────────────────────────────┐
│       Simulation & Exploration Layer (What-if)              │
│  • Modify model/GPU → see cost/latency impact               │
│  • Compare scenarios side-by-side                           │
│  • Adjust SLOs → see viable configurations                  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ↓
          User confirms configuration
                        │
                        ↓
┌─────────────────────────────────────────────────────────────┐
│         Deployment Automation Engine                        │
│  → Generate KServe/vLLM YAML                                │
│  → Configure autoscaling                                    │
│  → Deploy to Kubernetes                                     │
│  → Setup observability hooks                                │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ↓
      Return: deployment ID + monitoring links + cost estimate
                        │
                        ↓ (Post-deployment feedback loop)
┌─────────────────────────────────────────────────────────────┐
│         Knowledge Base ← Deployment Outcomes                │
│  Store: actual TTFT/TPOT, cost, SLO compliance              │
│  Use for: improving future recommendations                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Phase 1 (3-Month) Minimal Architecture

For the initial iteration, recommended technology choices:

| Component | Technology Choice | Rationale |
|-----------|------------------|-----------|
| **Conversational UI** | Chainlit | Fastest time-to-value for LLM apps |
| **Intent Extraction** | LangChain + Pydantic | Structured output with validation |
| **Recommendation Engine** | Rule-based + LLM augmentation | Explainable, no ML training needed |
| **Simulation** | Analytical formulas | Simple cost/latency estimates |
| **Deployment Automation** | Jinja2 + Kubernetes Python client | Direct control, no GitOps overhead yet |
| **Knowledge Base** | PostgreSQL + JSON columns | Familiar, good enough for MVP |
| **LLM Backend** | Self-hosted Llama 3 via vLLM | Dogfooding, cost-effective |
| **Orchestration** | FastAPI + in-memory state | Minimal complexity |
| **Observability** | OpenTelemetry + Prometheus | Standard, well-integrated |

---

## Key Design Principles

1. **Conversational-first**: All complexity hidden behind natural language (no forms)
2. **Benchmarking-driven**: Recommendations grounded in real performance data
3. **Explainable**: Users understand *why* each recommendation is made
4. **Iterative**: Support what-if exploration before committing resources
5. **One-click deployment**: Minimize manual YAML editing
6. **Observability by default**: Every deployment includes monitoring hooks

---

## Integration Points

### With RHAI Ecosystem
- **KServe**: Primary model serving platform
- **vLLM**: LLM runtime for inference
- **OpenShift**: Kubernetes platform
- **InstructLab**: Model fine-tuning and serving
- **RHAI Observability**: Metrics and monitoring infrastructure

### External Dependencies
- Model registries (HuggingFace Hub, custom)
- GPU hardware inventory systems
- Cost management APIs
- Identity/authentication (SSO)

---

## Security Considerations

1. **Input Validation**: Sanitize all user inputs before LLM processing
2. **Configuration Validation**: Verify generated YAML before Kubernetes deployment
3. **RBAC**: Respect cluster-level permissions for deployments
4. **Secrets Management**: Never expose credentials in configs or logs
5. **Audit Trail**: Log all deployment actions with user attribution

---

## Success Metrics

### User Experience
- Time from conversation start to successful deployment (target: < 10 minutes)
- Recommendation acceptance rate (target: > 70%)
- User satisfaction score (target: > 4/5)

### System Performance
- Conversation response latency (target: < 2 seconds)
- Deployment success rate (target: > 95%)
- System uptime (target: 99.9%)

### Business Impact
- Reduction in overprovisioning (measure GPU utilization)
- Increase in AI adoption (measure deployments/month)
- Customer retention (measure engagement with assistant)

---

## Open Questions for Refinement

1. **Multi-tenancy**: How do we isolate conversations/deployments across users?
2. **Security**: How do we validate generated configs before K8s deployment?
3. **Feedback loop**: How do we capture post-deployment outcomes to improve recommendations?
4. **Model catalog integration**: Do we sync with HuggingFace, or maintain a curated list?
5. **Versioning**: How do we handle updates to recommendations as benchmarks improve?
6. **Rollback**: What's the UX for reverting a deployment if it doesn't meet expectations?
7. **Conversational clarification flow** (Future Phase): How do we handle insufficient information gracefully?
   - When user provides minimal details, system should ask targeted follow-up questions
   - More detailed information → more precise recommendations with narrower option set
   - Limited information → broader recommendation set with clear pros/cons for each
   - System should gracefully handle "I don't know" responses and still provide useful recommendations
   - Balance between gathering information and not overwhelming user with too many questions

---

## Future Enhancements (Post-Phase 1)

### Traffic Modeling (Phase 2)
- Full statistical distributions (not just point estimates)
- Advanced burstiness pattern recognition
- Request heterogeneity modeling (mix of short/long requests)
- Streaming vs non-streaming workload separation

### Capacity Planning (Phase 2)
- Advanced tensor parallelism optimization
- Pipeline parallelism support
- Auto-scaling policy generation (min/max replicas)
- Batch size optimization for throughput/latency trade-offs

### Day-2 Operations
- Proactive optimization recommendations for running deployments
- Automated quantization suggestions based on latency requirements
- Speculative decoding automation
- Cost anomaly detection and alerts
- Drift detection (actual vs predicted SLOs)

### Advanced Features
- Multi-model deployments (model ensembles)
- A/B testing automation
- Fine-tuning pipeline integration
- Capacity planning for MaaS deployments
- Continuous learning from deployment outcomes
- Quality metric integration (BLEU scores, accuracy benchmarks)

---

## References

- Project Proposal: RHAI Assistant Pitch (July 2025)
- Feature List: AI Deployment Assistant 1-N
- Initial Scope: Pre-Deployment Assistant for LLM Serving
