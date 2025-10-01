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
**Purpose**: Extract structured intent from conversational input (task type, load, constraints)

**Technology Options**:
- **LangChain with structured output** - Built-in prompt engineering for extraction
- **Guardrails AI** - Schema validation and structured extraction from LLMs
- **Custom LLM + Pydantic validators** - Full control with type-safe extraction

**Key Functions**:
- Parse user inputs into structured data (task definition, expected load, priorities)
- Validate completeness before proceeding to recommendations
- Store conversation context for iterative refinement

**Data Schema Example**:
```python
class DeploymentIntent:
    task_type: str  # "chatbot", "summarization", "code-completion"
    expected_qps: float
    concurrent_users: int
    priority: str  # "accuracy", "latency", "cost"
    constraints: dict  # budget, max_latency, etc.
```

---

### 3. Recommendation Engine
**Purpose**: Suggest model, hardware, and SLO configurations based on captured context

**Technology Options**:
- **Rule-based decision tree + LLM augmentation** - Hybrid approach with explainability
- **Vector similarity search (FAISS/Pinecone) + retrieval** - Match context to known deployment patterns
- **Custom ML model** - Train on historical deployment data (future iteration)

**Inputs**:
- User context (task, load, constraints)
- Benchmarking data (from Knowledge Base)
- Hardware availability (from cluster inventory)

**Outputs**:
- Ranked model recommendations with rationale
- Hardware profiles (GPU/CPU types, counts)
- Suggested SLOs (latency, throughput targets)

**Recommendation Logic**:
1. Filter models by task compatibility
2. Estimate resource requirements based on expected load
3. Match to available hardware profiles
4. Rank by user priority (cost vs latency vs accuracy)
5. Generate explanation for each recommendation

---

### 4. Simulation & Exploration Layer
**Purpose**: Enable what-if analysis with cost/latency/throughput predictions

**Technology Options**:
- **Custom analytical models** - Formula-based estimation (tokens/sec × cost/token)
- **Discrete event simulation (SimPy)** - More accurate workload modeling
- **Historical performance extrapolation** - Use real benchmark data interpolation

**Key Functions**:
- Cost estimation (GPU hours × hourly rate)
- Latency prediction (model size + GPU memory bandwidth)
- Throughput calculation (batch size + concurrency limits)
- Scenario comparison UI (side-by-side configs)

**Simulation Outputs**:
- Estimated monthly cost
- P50/P95/P99 latency predictions
- Expected throughput (requests/sec)
- Resource utilization estimates

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
**Purpose**: Store performance data, industry standards, and deployment patterns

**Technology Options**:
- **PostgreSQL + pgvector** - Relational data + vector search for similarity matching
- **MongoDB + Atlas Search** - Flexible schema for varied benchmark data
- **Specialized vector DB (Qdrant/Weaviate)** - Optimized for embedding-based retrieval

**Data Schema**:
- Model benchmarks (throughput, latency, memory usage per GPU type)
- Hardware profiles (GPU specs, availability, cost)
- Deployment templates (curated configs for common scenarios)
- Historical deployment outcomes (for continuous learning)

**Benchmark Data Example**:
```json
{
  "model_id": "meta-llama/Llama-3-8b",
  "gpu_type": "NVIDIA-L4",
  "batch_size": 32,
  "metrics": {
    "throughput_tokens_per_sec": 1250,
    "latency_p50_ms": 45,
    "latency_p99_ms": 120,
    "memory_usage_gb": 16
  }
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

### 9. Observability & Telemetry
**Purpose**: Monitor assistant performance and user interactions

**Technology Options**:
- **OpenTelemetry + Prometheus + Grafana** - Standard observability stack
- **LangSmith** - LLM-specific tracing and debugging
- **Custom logging + ELK stack** - Flexible log aggregation

**Metrics to Track**:
- Conversation completion rate
- Recommendation acceptance rate
- Deployment success rate
- Time-to-deployment
- User satisfaction scores
- System performance (latency, error rates)

---

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                         User                                │
└───────────────────────┬─────────────────────────────────────┘
                        │ Natural Language Input
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              Conversational Interface Layer                 │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              Context & Intent Engine                        │
│              (Extract structured requirements)              │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ↓
┌─────────────────────────────────────────────────────────────┐
│              Recommendation Engine                          │
│              ← Query Knowledge Base                         │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ↓
         Present recommendations to user
                        │
                        ↓
┌─────────────────────────────────────────────────────────────┐
│         Simulation Layer (What-if exploration)              │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ↓
          User confirms configuration
                        │
                        ↓
┌─────────────────────────────────────────────────────────────┐
│         Deployment Automation Engine                        │
│         → Generate YAML → Deploy to Kubernetes              │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ↓
      Return deployment ID + observability links
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

---

## Future Enhancements (Post-Phase 1)

### Day-2 Operations
- Proactive optimization recommendations for running deployments
- Automated quantization suggestions
- Speculative decoding automation
- Cost anomaly detection

### Advanced Features
- Multi-model deployments (model ensembles)
- A/B testing automation
- Fine-tuning pipeline integration
- Capacity planning for MaaS deployments
- Continuous learning from deployment outcomes

---

## References

- Project Proposal: RHAI Assistant Pitch (July 2025)
- Feature List: AI Deployment Assistant 1-N
- Initial Scope: Pre-Deployment Assistant for LLM Serving
