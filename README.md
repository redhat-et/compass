# AI Pre-Deployment Assistant - POC

A conversational AI system that guides users from concept to production-ready LLM deployments through intelligent capacity planning and SLO-driven recommendations.

## Overview

This is a **Phase 1 POC** demonstrating the core architecture of the Pre-Deployment Assistant. The system uses:
- **Conversational AI** to extract deployment requirements from natural language
- **SLO-driven capacity planning** to recommend optimal model + GPU configurations
- **What-if analysis** to explore trade-offs between cost, latency, and throughput
- **YAML generation** for KServe/vLLM deployment configurations

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed system design.

## Project Structure

```
ai-assistant/
├── backend/                    # FastAPI backend with core recommendation engine
│   └── src/
│       ├── context_intent/     # Component 2: Intent extraction
│       ├── recommendation/     # Component 3: Model/GPU recommendations & capacity planning
│       ├── simulation/         # Component 4: What-if analysis (placeholder)
│       ├── deployment/         # Component 5: YAML generation & K8s deployment
│       ├── knowledge_base/     # Component 6: Data access layer (benchmarks, catalog, SLOs)
│       ├── orchestration/      # Component 8: Workflow coordination
│       ├── llm/                # Component 7: Ollama LLM client
│       └── api/                # FastAPI REST endpoints
├── ui/                         # Component 1: Streamlit conversational UI
├── data/                       # Synthetic data for POC (benchmarks, catalog, SLO templates)
├── generated_configs/          # Output directory for generated YAML files
├── config/                     # Configuration files (KIND cluster)
├── scripts/                    # Automation scripts (run_api.sh, run_ui.sh, kcluster.sh)
├── tests/                      # Test suites (sprint2, sprint4, sprint5)
├── docs/                       # Architecture documentation and diagrams
├── CLAUDE.md                   # AI assistant guidance
└── README.md                   # This file
```

## Prerequisites

### Required for All Features

- **Python 3.11+**
- **Ollama** installed and running
  ```bash
  brew install ollama
  ollama serve  # Start Ollama service
  ```

### Required for Sprint 5+ (Kubernetes Deployment)

- **Docker Desktop** - Required for KIND
  ```bash
  # Download from https://www.docker.com/products/docker-desktop
  # Or install via Homebrew
  brew install --cask docker
  ```

- **kubectl** - Kubernetes CLI
  ```bash
  brew install kubectl
  ```

- **KIND (Kubernetes in Docker)** - Will be installed automatically by setup scripts
  ```bash
  brew install kind
  ```

## Setup Instructions

**Note:** Backend and frontend have separate virtual environments. Set them up in separate terminal sessions or deactivate between steps.

### 1. Install Backend Dependencies

**Terminal 1:**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Verify correct venv
which python  # Should show: .../backend/venv/bin/python

pip install -r requirements.txt
```

### 2. Install Frontend Dependencies

**Terminal 2 (or deactivate first):**
```bash
# If in same terminal: deactivate first
deactivate

cd frontend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Verify correct venv
which python  # Should show: .../frontend/venv/bin/python

pip install -r requirements.txt
```

### 3. Pull Ollama Model

The POC uses `llama3.1:8b` for intent extraction:

```bash
ollama pull llama3.1:8b
```

**Alternative models** (if needed):
- `llama3.2:3b` - Smaller/faster, less accurate
- `mistral:7b` - Good balance of speed and quality

### 4. Verify Ollama Setup

```bash
# Test Ollama is working
ollama list  # Should show llama3.1:8b
```

## Running the POC

### Option 1: Run Full Stack with UI (Sprint 3 - Recommended)

The easiest way to use the assistant:

```bash
# Terminal 1 - Start Ollama (if not already running)
ollama serve

# Terminal 2 - Start FastAPI Backend
scripts/run_api.sh

# Terminal 3 - Start Streamlit UI
scripts/run_ui.sh
```

Then open http://localhost:8501 in your browser.

### Option 2: Test End-to-End Workflow

Test the complete recommendation workflow with demo scenarios:

```bash
cd backend
source venv/bin/activate
python test_workflow.py
```

This tests all 3 demo scenarios end-to-end.

### Option 3: Run FastAPI Backend Only

Start the API server:

```bash
./run_api.sh
```

Or manually:

```bash
scripts/run_api.sh
```

Test the API:

```bash
# Health check
curl http://localhost:8000/health

# Full recommendation
curl -X POST http://localhost:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"message": "I need a chatbot for 5000 users with low latency"}'
```

### Option 4: Test Individual Components

Test the LLM client:

```bash
cd backend
source venv/bin/activate
python -c "
from src.llm.ollama_client import OllamaClient
client = OllamaClient(model='llama3.2:3b')
print('Ollama available:', client.is_available())
print('Pulling model...')
client.ensure_model_pulled()
print('Model ready!')
"
```

## Demo Scenarios

The POC includes 3 pre-configured scenarios (see [data/demo_scenarios.json](data/demo_scenarios.json)):

1. **Customer Service Chatbot** - High volume (5000 users), strict latency (<500ms)
   - Expected: Llama 3.1 8B on 2x A100-80GB

2. **Code Generation Assistant** - Developer team (500 users), quality > speed
   - Expected: Llama 3.1 70B on 4x A100-80GB (tensor parallel)

3. **Document Summarization** - Batch processing (2000 users/day), cost-sensitive
   - Expected: Mistral 7B on 2x A10G

## Synthetic Data

All data files in `data/` are synthetic for POC purposes:

- **benchmarks.json**: 24 benchmark entries covering 10 models × 4 GPU types
- **slo_templates.json**: 7 use case templates with SLO targets
- **model_catalog.json**: 10 approved models with metadata
- **sample_outcomes.json**: 5 mock deployment outcomes

## Architecture Components (Sprint Roadmap)

- ✅ **Sprint 1** (Complete): Project structure, synthetic data, LLM client
- ✅ **Sprint 2** (Complete): Intent extraction, recommendation engines, knowledge base, FastAPI backend
- ✅ **Sprint 3** (Complete): Streamlit UI with chat and spec editor
- ✅ **Sprint 4** (Complete): YAML generation (KServe/vLLM/HPA/ServiceMonitor), mock monitoring dashboard
- ✅ **Sprint 5** (Complete): KIND cluster setup, KServe installation, actual Kubernetes deployment
- ⏳ **Sprint 6**: End-to-end deployment with real model inference

## Key Technologies

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI, Pydantic |
| Frontend | Streamlit |
| LLM | Ollama (llama3.1:8b) |
| Data | JSON files (Phase 1), PostgreSQL (Phase 2+) |
| YAML Generation | Jinja2 templates |
| Kubernetes | KIND (local), KServe v0.13.0 |
| Deployment | kubectl, Kubernetes Python client |

## Configuration

Edit `backend/src/llm/ollama_client.py` to change the default model:

```python
def __init__(self, model: str = "llama3.2:3b", host: Optional[str] = None):
```

## Troubleshooting

### Ollama Connection Issues

```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# If not running
ollama serve
```

### Model Not Found

```bash
ollama pull llama3.2:3b
```

### Import Errors

```bash
# Make sure you're in the right venv
which python  # Should show path to venv

# Reinstall dependencies
pip install -r requirements.txt
```

## Testing

See [backend/TESTING.md](backend/TESTING.md) for detailed testing instructions.

Quick tests:

```bash
# Test end-to-end workflow
cd backend && source venv/bin/activate
cd ..
python tests/test_sprint2.py

# Test Sprint 4: YAML generation and validation
python tests/test_sprint4.py

# Test Sprint 5: Kubernetes deployment
python tests/test_sprint5.py

# Test FastAPI endpoints
scripts/run_api.sh  # Start server in terminal 1
# In terminal 2:
curl -X POST http://localhost:8000/api/v1/test
```

## Sprint 4 Features (NEW!)

Sprint 4 is complete! New features:

### YAML Deployment Generation
- ✅ KServe InferenceService YAML with vLLM configuration
- ✅ HorizontalPodAutoscaler (HPA) for autoscaling
- ✅ Prometheus ServiceMonitor for metrics collection
- ✅ Grafana Dashboard ConfigMap
- ✅ Full YAML validation before generation
- ✅ Files written to `generated_configs/` directory

### Mock Monitoring Dashboard
- ✅ **Monitoring tab** in Streamlit UI showing Component 9 (Inference Observability)
- ✅ **SLO Compliance** metrics (TTFT, TPOT, E2E latency, throughput, uptime)
- ✅ **Resource Utilization** (GPU usage, memory, batch size, queue depth)
- ✅ **Cost Analysis** (actual vs predicted monthly cost, per-token cost)
- ✅ **Traffic Patterns** (actual vs predicted prompt/generation tokens, QPS)
- ✅ **Optimization Recommendations** (auto-generated suggestions)

### How to Use Sprint 4 Features
1. Get a deployment recommendation from the chat interface
2. Go to the **Cost** tab and click **"Generate Deployment YAML"**
3. View generated YAML file paths
4. Go to the **Monitoring** tab to see simulated observability dashboard
5. Check `generated_configs/` directory for all YAML files

## Sprint 5 Features (NEW!)

Sprint 5 is complete! New Kubernetes deployment capabilities:

### KIND Cluster Setup
- ✅ Local Kubernetes cluster running in Docker
- ✅ 3-node cluster with simulated GPU labels (A100-80GB, L4)
- ✅ KServe v0.13.0 installed and configured
- ✅ cert-manager v1.14.4 for certificate management

### Actual Kubernetes Deployment
- ✅ **Deploy to Cluster** button now functional in UI
- ✅ Auto-detects cluster accessibility
- ✅ Deploys InferenceService and HPA resources to cluster
- ✅ Real-time deployment status from Kubernetes
- ✅ Pod and resource monitoring in UI

### How to Use Sprint 5 Features

#### 1. Set Up KIND Cluster (One-time setup)

**Easy Way (Recommended):**
```bash
# Ensure Docker Desktop is running

# Create and configure cluster (installs KIND, cert-manager, KServe)
scripts/kcluster.sh start
```

**Manual Way:**
```bash
# Ensure Docker Desktop is running

# Install KIND (if not already installed)
brew install kind

# Create cluster with KServe
kind create cluster --config config/kind-cluster.yaml

# Install cert-manager
kubectl apply -f https://github.com/cert-manager/cert-manager/releases/download/v1.14.4/cert-manager.yaml

# Wait for cert-manager
kubectl wait --for=condition=available --timeout=300s -n cert-manager deployment/cert-manager

# Install KServe
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.13.0/kserve.yaml
kubectl apply -f https://github.com/kserve/kserve/releases/download/v0.13.0/kserve-cluster-resources.yaml

# Wait for KServe
kubectl wait --for=condition=available --timeout=300s -n kserve deployment/kserve-controller-manager

# Configure KServe for RawDeployment mode
kubectl patch configmap/inferenceservice-config -n kserve --type=strategic -p '{"data": {"deploy": "{\"defaultDeploymentMode\": \"RawDeployment\"}"}}'
```

#### 2. Deploy Models Through UI
1. Get a deployment recommendation from the chat interface
2. Click **"Generate Deployment YAML"** in the Actions section
3. If cluster is accessible, click **"Deploy to Kubernetes"**
4. Go to **Monitoring** tab to see:
   - Real Kubernetes deployment status
   - InferenceService conditions
   - Pod information
   - Simulated performance metrics

#### 3. Test End-to-End Deployment
```bash
# Run automated test suite
cd backend
source venv/bin/activate
cd ..
python tests/test_sprint5.py
```

### Cluster Management

The `kcluster.sh` script provides easy cluster lifecycle management:

**Check cluster status:**
```bash
scripts/kcluster.sh status
```

**Restart cluster (fresh start):**
```bash
scripts/kcluster.sh restart
```

**Stop/delete cluster:**
```bash
scripts/kcluster.sh stop
```

**View help:**
```bash
scripts/kcluster.sh help
```

**Manual kubectl commands:**
```bash
# View all resources
kubectl get pods -A

# View deployments
kubectl get inferenceservices
kubectl get pods

# Delete a specific deployment
kubectl delete inferenceservice <deployment-id>

# Check cluster info
kubectl cluster-info
```

## Next Steps

Sprint 5 complete! Next up (Sprint 6):
1. Deploy actual vLLM models to cluster
2. Real model inference testing
3. Performance validation
4. End-to-end observability

## Contributing

This is a POC/design phase repository. See [CLAUDE.md](CLAUDE.md) for AI assistant guidance when making changes.

## License

[To be determined]
