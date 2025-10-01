# Pre-Deployment Assistant Architecture Diagram

```mermaid
graph TB
    subgraph "User Interface"
        UI[Conversational Interface Layer<br/>Chainlit/Streamlit/React]
    end

    subgraph "Core Processing Layer"
        CIE[Context & Intent Engine<br/>LangChain + Pydantic]
        RE[Recommendation Engine<br/>Rule-based + LLM]
        SIM[Simulation & Exploration<br/>Analytical Models]
        LLM[LLM Backend<br/>Llama 3 via vLLM]
    end

    subgraph "Deployment Layer"
        DAE[Deployment Automation Engine<br/>Jinja2 + K8s Client]
        ORCH[Orchestration Engine<br/>FastAPI + State Mgmt]
    end

    subgraph "Data Layer"
        KB[(Knowledge Base<br/>PostgreSQL + pgvector)]
    end

    subgraph "External Systems"
        K8S[Kubernetes/OpenShift Cluster]
        KSERVE[KServe API]
        VLLM[vLLM Runtime]
        REGISTRY[Model Registry<br/>HuggingFace Hub]
        OBS[Observability Stack<br/>Prometheus + Grafana]
    end

    subgraph "Monitoring"
        TEL[Telemetry & Observability<br/>OpenTelemetry]
    end

    %% User interaction flow
    UI -->|Natural Language Input| CIE
    CIE -->|Structured Intent| RE

    %% LLM integration
    CIE <-.->|Extract Intent| LLM
    RE <-.->|Generate Recommendations| LLM

    %% Recommendation flow
    RE -->|Query Benchmarks| KB
    RE -->|Recommendations| SIM
    SIM -->|What-if Results| UI

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
    RE -.->|Recommendation Metrics| TEL
    DAE -.->|Deployment Metrics| TEL
    K8S -.->|Cluster Metrics| TEL
    TEL -->|Store Metrics| KB

    %% Feedback loop
    OBS -.->|Performance Data| KB
    K8S -.->|Deployment Outcomes| KB

    style UI fill:#e1f5ff
    style CIE fill:#fff4e1
    style RE fill:#fff4e1
    style SIM fill:#fff4e1
    style LLM fill:#ffe1f5
    style DAE fill:#e1ffe1
    style ORCH fill:#e1ffe1
    style KB fill:#f0f0f0
    style TEL fill:#ffe1e1
```

## Component Flow Description

### 1. User Interaction Flow
1. User interacts via **Conversational Interface** (Chainlit/Streamlit)
2. **Context & Intent Engine** extracts structured requirements using LLM
3. **Recommendation Engine** generates model/hardware/SLO suggestions
4. User explores options via **Simulation Layer**
5. User confirms configuration

### 2. Deployment Flow
1. **Deployment Automation Engine** generates KServe/vLLM configs
2. **Orchestration Engine** coordinates deployment steps
3. Manifests deployed to **Kubernetes/OpenShift**
4. **KServe** provisions model serving endpoint
5. **vLLM** runtime configured for inference
6. **Observability** hooks configured automatically

### 3. Data Flow
- **Knowledge Base** stores benchmarks, hardware profiles, templates
- **LLM Backend** powers conversational AI and recommendations
- **Telemetry** captures user interactions, deployment outcomes, system metrics
- **Feedback Loop** enriches Knowledge Base with real deployment data

### 4. Integration Points
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
    participant CIE as Context Engine
    participant LLM as LLM Backend
    participant RE as Recommendation Engine
    participant KB as Knowledge Base
    participant SIM as Simulation Layer
    participant DAE as Deployment Engine
    participant K8S as Kubernetes

    User->>UI: "I need a chatbot for 1000 users"
    UI->>CIE: Parse intent
    CIE->>LLM: Extract structured data
    LLM-->>CIE: {task: chatbot, users: 1000, priority: latency}

    CIE->>RE: Get recommendations
    RE->>KB: Query benchmarks
    KB-->>RE: Model performance data
    RE->>LLM: Generate recommendations
    LLM-->>RE: Ranked suggestions with rationale
    RE-->>UI: Display recommendations

    User->>UI: "What if I use cheaper GPU?"
    UI->>SIM: Simulate scenario
    SIM->>KB: Get GPU pricing
    SIM-->>UI: Cost/latency comparison

    User->>UI: "Deploy with L4 GPU"
    UI->>DAE: Generate config
    DAE->>KB: Get deployment template
    DAE->>DAE: Generate KServe YAML
    DAE->>K8S: Deploy InferenceService
    K8S-->>DAE: Deployment ID
    DAE-->>UI: Success + monitoring links
    UI-->>User: "Deployed successfully!"
```

---

## State Machine - Workflow Orchestration

```mermaid
stateDiagram-v2
    [*] --> ContextCollection

    ContextCollection --> ValidationPending: User provides info
    ValidationPending --> ContextCollection: Incomplete/Invalid
    ValidationPending --> RecommendationGeneration: Valid & Complete

    RecommendationGeneration --> PresentingOptions: Recommendations ready
    PresentingOptions --> InteractiveExploration: User explores
    PresentingOptions --> ConfigurationFinalized: User confirms

    InteractiveExploration --> PresentingOptions: Update parameters

    ConfigurationFinalized --> DeploymentExecution: Generate configs
    DeploymentExecution --> PostDeploymentValidation: Deployment triggered
    DeploymentExecution --> Failed: Deployment error

    PostDeploymentValidation --> Success: Health checks pass
    PostDeploymentValidation --> Failed: Health checks fail

    Failed --> [*]
    Success --> [*]
```

---

## Data Model - Knowledge Base Schema

```mermaid
erDiagram
    MODEL_BENCHMARKS ||--o{ HARDWARE_PROFILES : "tested_on"
    MODEL_BENCHMARKS ||--o{ DEPLOYMENT_TEMPLATES : "references"
    DEPLOYMENT_OUTCOMES }o--|| DEPLOYMENT_TEMPLATES : "uses"
    HARDWARE_PROFILES ||--o{ COST_DATA : "has"

    MODEL_BENCHMARKS {
        string model_id PK
        string gpu_type FK
        int batch_size
        float throughput_tokens_per_sec
        float latency_p50_ms
        float latency_p99_ms
        float memory_usage_gb
        timestamp created_at
    }

    HARDWARE_PROFILES {
        string gpu_type PK
        int vram_gb
        float tflops
        string manufacturer
        boolean available
        timestamp updated_at
    }

    DEPLOYMENT_TEMPLATES {
        string template_id PK
        string task_type
        string model_id FK
        string gpu_type FK
        json kserve_config
        json vllm_config
        json autoscaling_config
        timestamp created_at
    }

    DEPLOYMENT_OUTCOMES {
        string deployment_id PK
        string template_id FK
        float actual_latency_p50_ms
        float actual_throughput
        float cost_per_hour
        boolean meets_slo
        timestamp deployed_at
    }

    COST_DATA {
        string gpu_type FK
        string cloud_provider
        float hourly_rate_usd
        timestamp effective_date
    }
```

---

## Technology Stack Summary

| Layer | Component | Technology |
|-------|-----------|------------|
| **Presentation** | UI Framework | Chainlit |
| **Application** | Intent Extraction | LangChain + Pydantic |
| **Application** | Recommendations | Rule-based + LLM |
| **Application** | Simulation | Analytical formulas |
| **Application** | Orchestration | FastAPI |
| **AI/ML** | LLM Backend | Llama 3 via vLLM |
| **Data** | Knowledge Base | PostgreSQL + pgvector |
| **Deployment** | Config Generation | Jinja2 |
| **Deployment** | K8s Integration | Kubernetes Python Client |
| **Observability** | Metrics | OpenTelemetry + Prometheus |
| **Observability** | Dashboards | Grafana |
| **Infrastructure** | Container Platform | OpenShift/Kubernetes |
| **Infrastructure** | Model Serving | KServe + vLLM |
