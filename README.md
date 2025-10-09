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
├── simulator/                  # vLLM simulator service for GPU-free development
├── config/                     # Configuration files (KIND cluster)
├── scripts/                    # Automation scripts (run_api.sh, run_ui.sh, kind-cluster.sh)
├── tests/                      # Test suites
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

### Required for Kubernetes Deployment

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

### Option 1: Run Full Stack with UI (Recommended)

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

## Implemented Features

- ✅ **Foundation**: Project structure, synthetic data, LLM client (Ollama)
- ✅ **Core Recommendation Engine**: Intent extraction, traffic profiling, model recommendation, capacity planning
- ✅ **FastAPI Backend**: REST endpoints, orchestration workflow, knowledge base access
- ✅ **Streamlit UI**: Chat interface, recommendation display, specification editor
- ✅ **Deployment Automation**: YAML generation (KServe/vLLM/HPA/ServiceMonitor), Kubernetes deployment
- ✅ **Local Kubernetes**: KIND cluster support, KServe installation, cluster management
- ✅ **vLLM Simulator**: GPU-free development mode with realistic latency simulation
- ✅ **Monitoring & Testing**: Real-time deployment status, inference testing UI, cluster observability

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
python tests/test_workflow.py

# Test FastAPI endpoints
scripts/run_api.sh  # Start server in terminal 1
# In terminal 2:
curl -X POST http://localhost:8000/api/v1/test
```

## YAML Deployment Generation

The system automatically generates production-ready Kubernetes configurations:

- ✅ KServe InferenceService YAML with vLLM configuration
- ✅ HorizontalPodAutoscaler (HPA) for autoscaling
- ✅ Prometheus ServiceMonitor for metrics collection
- ✅ Grafana Dashboard ConfigMap
- ✅ Full YAML validation before generation
- ✅ Files written to `generated_configs/` directory

**How to use:**
1. Get a deployment recommendation from the chat interface
2. Go to the **Cost** tab and click **"Generate Deployment YAML"**
3. View generated YAML file paths
4. Check `generated_configs/` directory for all YAML files

## Kubernetes Deployment

The system supports deploying to local Kubernetes clusters:

- ✅ Local KIND cluster running in Docker
- ✅ 3-node cluster with simulated GPU labels (A100-80GB, L4)
- ✅ KServe v0.13.0 installed and configured
- ✅ cert-manager v1.14.4 for certificate management
- ✅ **Deploy to Cluster** button in UI
- ✅ Auto-detects cluster accessibility
- ✅ Deploys InferenceService and HPA resources to cluster
- ✅ Real-time deployment status from Kubernetes
- ✅ Pod and resource monitoring in UI

### Set Up KIND Cluster (One-time setup)

**Easy Way (Recommended):**
```bash
# Ensure Docker Desktop is running

# Create and configure cluster (installs KIND, cert-manager, KServe)
scripts/kind-cluster.sh start
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

### Deploy Models Through UI
1. Get a deployment recommendation from the chat interface
2. Click **"Generate Deployment YAML"** in the Actions section
3. If cluster is accessible, click **"Deploy to Kubernetes"**
4. Go to **Monitoring** tab to see:
   - Real Kubernetes deployment status
   - InferenceService conditions
   - Pod information
   - Performance metrics

### Cluster Management

The `kind-cluster.sh` script provides easy cluster lifecycle management:

**Check cluster status:**
```bash
scripts/kind-cluster.sh status
```

**Restart cluster (fresh start):**
```bash
scripts/kind-cluster.sh restart
```

**Stop/delete cluster:**
```bash
scripts/kind-cluster.sh stop
```

**View help:**
```bash
scripts/kind-cluster.sh help
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

## vLLM Simulator (GPU-Free Development)

The system includes a vLLM simulator for development without GPU hardware:

**Simulator Features:**
- ✅ **GPU-free development** - Run deployments on any laptop without GPU hardware
- ✅ **OpenAI-compatible API** - Implements `/v1/completions` and `/v1/chat/completions`
- ✅ **Realistic performance** - Uses benchmark data to simulate TTFT/TPOT latency
- ✅ **Pattern-based responses** - Returns appropriate canned responses (code, chat, summarization, etc.)
- ✅ **Single Docker image** - Configure model via environment variables
- ✅ **Prometheus metrics** - Exposes `/metrics` endpoint matching vLLM format

**Inference Testing:**
- ✅ **Test deployed models** directly from the Monitoring tab
- ✅ **Send custom prompts** and see responses
- ✅ **View latency metrics** and token usage
- ✅ **Automatic port-forwarding** to Kubernetes service

### Deploy a Model in Simulator Mode (default)

Simulator mode is enabled by default for all deployments:

```bash
# Start the UI
scripts/run_ui.sh

# In the UI:
# 1. Get a deployment recommendation
# 2. Click "Generate Deployment YAML"
# 3. Click "Deploy to Kubernetes"
# 4. Go to Monitoring tab
# 5. Pod should become Ready in ~10-15 seconds
```

### Test Inference

Once deployed:
1. Go to **Monitoring** tab
2. See "🧪 Inference Testing" section
3. Enter a test prompt
4. Click "🚀 Send Test Request"
5. View the simulated response and metrics

### Switch to Real vLLM (Future)

To use real vLLM with actual GPUs (Phase 2):
```python
# In backend/src/api/routes.py
deployment_generator = DeploymentGenerator(simulator_mode=False)
```

Then deploy to a GPU-enabled cluster.

### Simulator vs Real vLLM

| Feature | Simulator Mode | Real vLLM Mode |
|---------|---------------|----------------|
| GPU Required | ❌ No | ✅ Yes |
| Model Download | ❌ No | ✅ Yes (from HuggingFace) |
| Inference | Canned responses | Real generation |
| Latency | Simulated (from benchmarks) | Actual GPU performance |
| Use Case | Development, testing, demos | Production deployment |
| Cluster | Works on KIND (local) | Requires GPU-enabled cluster |

## Future Enhancements

Potential improvements for Phase 2:
1. **External Access for Production**:
   - Generate Ingress/IngressRoute YAML for permanent external access
   - Support multiple simultaneous services (multi-tenant deployments)
   - Path-based routing (e.g., `/inference/customer-service`)
   - Host-based routing (e.g., `customer-service.inference.company.com`)
   - TLS, authentication, and rate limiting at ingress layer
2. Streaming response support in simulator
3. Error injection for testing resilience
4. Real vLLM deployment mode with GPU validation
5. Collect actual performance metrics from real deployments
6. Feedback loop: actual metrics → benchmark updates
7. Statistical distributions for traffic (not just point estimates)
8. Multi-dimensional benchmarks (batch size, sequence length variations)
9. Security hardening (YAML validation, RBAC)
10. Multi-tenancy support (namespaces, resource quotas, network policies)

## Contributing

This is a POC/design phase repository. See [CLAUDE.md](CLAUDE.md) for AI assistant guidance when making changes.

## License

[To be determined]
