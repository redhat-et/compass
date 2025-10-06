# Pre-Deployment Assistant Architecture Diagram

```mermaid
graph TB
    subgraph "User Interface"
        UI[Conversational Interface Layer<br/>Streamlit]
        SPEC_UI[Specification Review UI<br/>Editable Deployment Spec]
    end

    subgraph "Core Processing Layer"
        CIE[Context & Intent Engine<br/>Extract Intent + Generate Spec]
        TPG[Traffic Profile Generator<br/>QPS, Prompt/Gen Lengths]
        MRE[Model Recommendation Engine<br/>Filter & Rank Models]
        CPE[Capacity Planning Engine<br/>GPU Count, Tensor Parallelism]
        SIM[Simulation & Exploration<br/>What-if Analysis]
        LLM[LLM Backend<br/>Ollama llama3.1:8b]
    end

    subgraph "Deployment Layer"
        DAE[Deployment Automation Engine<br/>Jinja2 + K8s Client]
        ORCH[Orchestration Engine<br/>FastAPI + State Mgmt]
    end

    subgraph "Data Layer"
        KB[(Knowledge Base<br/>JSON files POC → PostgreSQL Prod)]
        KB_BENCH[Model Benchmarks<br/>TTFT, TPOT, Throughput]
        KB_SLO[Use Case SLO Templates<br/>Default Targets]
        KB_MODELS[Model Catalog<br/>Curated Models]
        KB_OUTCOMES[Deployment Outcomes<br/>Historical Data]
    end

    subgraph "External Systems"
        K8S[Kubernetes/OpenShift Cluster]
        KSERVE[KServe API]
        VLLM[vLLM Runtime / Simulator]
        REGISTRY[Model Registry<br/>HuggingFace Hub]
        OBS[Observability Stack<br/>Prometheus + Grafana]
    end

    subgraph "Development Tools"
        SIM_VLLM[vLLM Simulator<br/>GPU-free Development]
    end

    subgraph "Monitoring"
        TEL[Telemetry & Observability<br/>OpenTelemetry]
    end

    %% User interaction flow
    UI -->|Natural Language Input| CIE
    CIE -->|Deployment Spec| SPEC_UI
    SPEC_UI -->|Confirmed/Edited Spec| TPG

    %% LLM integration
    CIE <-.->|Extract Intent| LLM
    MRE <-.->|Generate Recommendations| LLM

    %% Knowledge Base queries
    CIE -->|Lookup Use Case| KB_SLO
    TPG -->|Query Defaults| KB_SLO
    MRE -->|Query Models| KB_MODELS
    MRE -->|Query Benchmarks| KB_BENCH
    CPE -->|Query Performance| KB_BENCH

    %% Recommendation flow
    TPG -->|Traffic Profile| MRE
    TPG -->|Traffic Profile| CPE
    MRE -->|Model Candidates| CPE
    CPE -->|GPU Configs + Costs| SIM
    SIM -->|Recommendations| UI

    %% What-if exploration
    UI -->|Modify Params| SIM
    SIM -->|Updated Predictions| UI

    %% Deployment flow
    SIM -->|Confirmed Config| DAE
    DAE -->|Generate Manifests| ORCH
    ORCH -->|Deploy| K8S
    ORCH -->|Configure| KSERVE
    ORCH -->|Setup Runtime| VLLM

    %% Model registry
    DAE -->|Fetch Models| REGISTRY

    %% Observability
    ORCH -->|Configure Monitoring| OBS
    UI -.->|User Metrics| TEL
    MRE -.->|Recommendation Metrics| TEL
    DAE -.->|Deployment Metrics| TEL
    K8S -.->|Cluster Metrics| TEL
    TEL -->|Store Metrics| KB

    %% Feedback loop
    OBS -.->|Actual TTFT/TPOT/Cost| KB_OUTCOMES
    K8S -.->|SLO Compliance| KB_OUTCOMES
    KB_OUTCOMES -.->|Improve Recommendations| MRE

    %% Knowledge Base internal connections
    KB_BENCH -.-> KB
    KB_SLO -.-> KB
    KB_MODELS -.-> KB
    KB_OUTCOMES -.-> KB

    style UI fill:#e1f5ff
    style SPEC_UI fill:#e1f5ff
    style CIE fill:#fff4e1
    style TPG fill:#fff4e1
    style MRE fill:#fff4e1
    style CPE fill:#fff4e1
    style SIM fill:#fff4e1
    style LLM fill:#ffe1f5
    style DAE fill:#e1ffe1
    style ORCH fill:#e1ffe1
    style KB fill:#f0f0f0
    style KB_BENCH fill:#f0f0f0
    style KB_SLO fill:#f0f0f0
    style KB_MODELS fill:#f0f0f0
    style KB_OUTCOMES fill:#f0f0f0
    style TEL fill:#ffe1e1
```

## Component Flow Description

### 1. User Interaction & Specification Generation
1. User interacts via **Conversational Interface** describing their use case
2. **Context & Intent Engine** extracts structured intent using LLM
3. Engine looks up **Use Case SLO Templates** for default targets
4. System generates complete **Deployment Specification** (traffic profile, SLOs, constraints)
5. **Specification Review UI** displays editable spec to user
6. User optionally modifies traffic params, SLO targets, or constraints

### 2. Recommendation Flow
1. **Traffic Profile Generator** converts high-level requirements → QPS, prompt/gen lengths
2. **Model Recommendation Engine** queries **Model Catalog** and filters by task compatibility
3. Engine queries **Model Benchmarks** for performance data
4. **Capacity Planning Engine** calculates GPU count and configuration for each model
5. System ranks recommendations by user priority (cost vs latency vs accuracy)
6. **Simulation Layer** presents ranked options with cost/latency/SLO predictions

### 3. What-If Exploration
1. User modifies model selection, GPU type, or SLO targets in **Simulation Layer**
2. System re-calculates capacity requirements and cost predictions
3. User compares scenarios side-by-side
4. User selects final configuration

### 4. Deployment Flow
1. **Deployment Automation Engine** generates KServe/vLLM configs from selected option
2. **Orchestration Engine** coordinates deployment steps
3. Manifests deployed to **Kubernetes/OpenShift**
4. **KServe** provisions model serving endpoint
5. **vLLM** runtime configured for inference
6. **Observability** hooks configured automatically

### 5. Feedback Loop
- **Observability Stack** captures actual TTFT, TPOT, throughput, cost
- **Deployment Outcomes** stored in Knowledge Base
- System learns from discrepancies between predicted vs actual performance
- Future recommendations improve based on historical data

### 6. Data Flow
- **Use Case SLO Templates** → Default targets for chatbot, summarization, etc.
- **Model Catalog** → Curated, approved models with metadata
- **Model Benchmarks** → TTFT/TPOT/throughput for model+GPU combinations
- **Deployment Outcomes** → Historical data for continuous learning
- **LLM Backend** powers conversational AI and recommendation explanations

### 7. Integration Points
- **Model Registry** (HuggingFace Hub) for model artifacts
- **Kubernetes API** for cluster operations
- **KServe API** for model serving
- **Prometheus/Grafana** for monitoring and alerting

---

## Detailed Architecture Diagram

```mermaid
C4Context
    title System Context - Pre-Deployment Assistant

    Person(dev, "Application Developer", "Wants to deploy LLM model")
    Person(admin, "Cluster Admin", "Manages infrastructure")

    System(assistant, "Pre-Deployment Assistant", "Guides users from concept to production")

    System_Ext(k8s, "Kubernetes/OpenShift", "Container orchestration")
    System_Ext(registry, "Model Registry", "HuggingFace Hub")
    System_Ext(obs, "Observability Platform", "Prometheus/Grafana")

    Rel(dev, assistant, "Describes deployment needs", "Natural Language")
    Rel(admin, assistant, "Plans capacity", "Natural Language")
    Rel(assistant, k8s, "Deploys models", "K8s API")
    Rel(assistant, registry, "Fetches models", "HTTP")
    Rel(assistant, obs, "Configures monitoring", "API")
    Rel(k8s, obs, "Sends metrics", "Prometheus")
```

---

## Component Interaction Sequence

```mermaid
sequenceDiagram
    participant User
    participant UI as Conversational UI
    participant SPEC_UI as Specification UI
    participant CIE as Context Engine
    participant LLM as LLM Backend
    participant TPG as Traffic Profile Gen
    participant MRE as Model Rec Engine
    participant CPE as Capacity Planner
    participant KB as Knowledge Base
    participant SIM as Simulation Layer
    participant DAE as Deployment Engine
    participant K8S as Kubernetes

    User->>UI: "I need a chatbot for 1000 users, low latency critical"
    UI->>CIE: Parse intent
    CIE->>LLM: Extract structured data
    LLM-->>CIE: {task: chatbot, users: 1000, priority: latency}

    CIE->>KB: Lookup use case "chatbot"
    KB-->>CIE: Default SLOs: TTFT<300ms, TPOT<50ms

    CIE->>SPEC_UI: Show generated spec
    SPEC_UI-->>User: Display: traffic profile, SLOs, budget
    User->>SPEC_UI: Edit TTFT target to 200ms
    SPEC_UI->>TPG: Confirmed spec

    TPG->>KB: Query traffic defaults
    KB-->>TPG: Avg prompt:150, gen:200 tokens
    TPG->>MRE: Traffic profile + SLO targets

    MRE->>KB: Query model catalog (task=chatbot)
    KB-->>MRE: Llama-3-8B, Llama-3-70B, Mistral-7B...
    MRE->>KB: Query benchmarks for candidates
    KB-->>MRE: TTFT/TPOT/throughput data

    MRE->>CPE: Model candidates + traffic + SLOs
    CPE->>KB: Query GPU costs
    CPE->>CPE: Calculate: Llama-3-8B needs 2x L4 for QPS
    CPE-->>SIM: Ranked recommendations

    SIM-->>UI: Option A: Llama-3-8B/2xL4/$800<br/>Option B: Llama-3-70B/4xA100/$2400

    User->>UI: "What if I use A100 instead?"
    UI->>SIM: Recalculate with A100 constraint
    SIM->>CPE: Re-run capacity calc
    CPE-->>SIM: Llama-3-8B/1xA100/$600, better TTFT
    SIM-->>UI: Updated prediction

    User->>UI: "Deploy Option A (2x L4)"
    UI->>DAE: Generate config for Llama-3-8B/2xL4
    DAE->>KB: Get deployment template
    DAE->>DAE: Generate KServe + vLLM YAML
    DAE->>K8S: Deploy InferenceService
    K8S-->>DAE: Deployment ID
    DAE-->>UI: Success + cost: $800/mo + monitoring link
    UI-->>User: "Deployed! Est TTFT: 185ms, TPOT: 42ms"
```

---

## State Machine - Workflow Orchestration

```mermaid
stateDiagram-v2
    [*] --> ContextCollection

    ContextCollection --> SpecificationGeneration: Intent extracted
    SpecificationGeneration --> SpecificationReview: Spec auto-generated

    SpecificationReview --> SpecificationEditing: User edits
    SpecificationReview --> TrafficProfileGeneration: User confirms
    SpecificationEditing --> TrafficProfileGeneration: Edits saved

    TrafficProfileGeneration --> ModelRecommendation: Traffic profile ready
    ModelRecommendation --> CapacityPlanning: Models filtered
    CapacityPlanning --> PresentingOptions: GPU configs calculated

    PresentingOptions --> InteractiveExploration: User explores what-if
    PresentingOptions --> ConfigurationFinalized: User confirms

    InteractiveExploration --> CapacityPlanning: Update params
    InteractiveExploration --> PresentingOptions: Predictions updated

    ConfigurationFinalized --> DeploymentExecution: Generate configs
    DeploymentExecution --> PostDeploymentValidation: Deployment triggered
    DeploymentExecution --> Failed: Deployment error

    PostDeploymentValidation --> ObservabilitySetup: Health checks pass
    PostDeploymentValidation --> Failed: Health checks fail

    ObservabilitySetup --> Success: Monitoring configured
    Success --> FeedbackLoop: Collect outcomes
    FeedbackLoop --> [*]

    Failed --> [*]
```

---

## Data Model - Knowledge Base Schema

```mermaid
erDiagram
    MODEL_BENCHMARKS ||--o{ HARDWARE_PROFILES : "tested_on"
    MODEL_BENCHMARKS }o--|| MODEL_CATALOG : "benchmarks"
    MODEL_CATALOG ||--o{ DEPLOYMENT_TEMPLATES : "uses"
    DEPLOYMENT_OUTCOMES }o--|| DEPLOYMENT_TEMPLATES : "instantiates"
    HARDWARE_PROFILES ||--o{ COST_DATA : "has_pricing"
    USE_CASE_SLO_TEMPLATES ||--o{ DEPLOYMENT_TEMPLATES : "guides"

    MODEL_BENCHMARKS {
        string model_id PK
        string gpu_type FK
        int tensor_parallel_degree
        string quantization
        json vllm_config
        float throughput_tokens_per_sec
        float throughput_requests_per_sec
        float ttft_p50_ms
        float ttft_p90_ms
        float ttft_p99_ms
        float tpot_p50_ms
        float tpot_p90_ms
        float tpot_p99_ms
        float e2e_latency_p50_ms
        float e2e_latency_p95_ms
        float e2e_latency_p99_ms
        float memory_usage_gb
        float gpu_utilization_pct
        json test_conditions
        timestamp created_at
    }

    HARDWARE_PROFILES {
        string gpu_type PK
        int vram_gb
        float tflops_fp16
        float memory_bandwidth_gbps
        string manufacturer
        string architecture
        boolean available
        json supported_parallelism
        int max_tensor_parallel
        timestamp updated_at
    }

    MODEL_CATALOG {
        string model_id PK
        string model_family
        string size_parameters
        int context_window
        json supported_tasks
        json subject_matter_domains
        string quality_tier
        string license
        boolean approved_for_production
        int minimum_vram_gb
        boolean supports_quantization
        timestamp created_at
    }

    USE_CASE_SLO_TEMPLATES {
        string use_case PK
        string description
        json default_slos
        json traffic_defaults
        json recommended_model_characteristics
        timestamp updated_at
    }

    DEPLOYMENT_TEMPLATES {
        string template_id PK
        string use_case FK
        string model_id FK
        string gpu_type FK
        int gpu_count
        int tensor_parallel_degree
        json kserve_config
        json vllm_config
        json autoscaling_config
        timestamp created_at
    }

    DEPLOYMENT_OUTCOMES {
        string deployment_id PK
        string template_id FK
        string model_id FK
        string gpu_type FK
        int gpu_count
        float ttft_p90_ms
        float tpot_p90_ms
        float e2e_latency_p95_ms
        float throughput_avg_rps
        float cost_per_hour_usd
        json slo_compliance
        json traffic_observed
        timestamp deployed_at
        int observation_period_days
    }

    COST_DATA {
        string gpu_type FK
        string cloud_provider
        string instance_type
        float hourly_rate_usd
        float monthly_rate_usd
        string region
        timestamp effective_date
    }
```

---

## Technology Stack Summary

| Layer | Component | Technology |
|-------|-----------|------------|
| **Presentation** | UI Framework | Streamlit (POC) |
| **Application** | Intent Extraction | Ollama (llama3.1:8b) + Pydantic |
| **Application** | Recommendations | Rule-based + LLM |
| **Application** | Simulation | Analytical formulas |
| **Application** | Orchestration | FastAPI |
| **AI/ML** | LLM Backend | Ollama (llama3.1:8b) |
| **Data** | Knowledge Base | JSON files (POC) → PostgreSQL (Prod) |
| **Deployment** | Config Generation | Jinja2 |
| **Deployment** | K8s Integration | Kubernetes Python Client |
| **Deployment** | vLLM Simulator | FastAPI + Docker |
| **Observability** | Metrics | Kubernetes API |
| **Observability** | Dashboards | Streamlit |
| **Infrastructure** | Container Platform | Kubernetes (KIND for POC) |
| **Infrastructure** | Model Serving | KServe + vLLM / Simulator |
