# Developer Guide

This guide provides step-by-step instructions for developing and testing Compass.

## Table of Contents

- [Development Environment Setup](#development-environment-setup)
- [Component Startup Sequence](#component-startup-sequence)
- [Development Workflows](#development-workflows)
- [Testing](#testing)
- [Debugging](#debugging)
- [Making Changes](#making-changes)

## Development Environment Setup

### Prerequisites

Ensure you have all required tools installed:

```bash
make check-prereqs
```

This checks for:
- Docker Desktop (running)
- Python 3.11+
- Ollama
- kubectl
- KIND

### Initial Setup

Create virtual environments and install dependencies:

```bash
make setup
```

This creates a single shared virtual environment in `venv/` (at project root) used by both the backend and UI.

## Component Startup Sequence

The system consists of 4 main components that must start in order:

### 1. Ollama Service

**Purpose:** LLM inference for intent extraction

**Start:**
```bash
make start-ollama
```

**Manual start:**
```bash
ollama serve
```

**Verify:**
```bash
curl http://localhost:11434/api/tags
ollama list  # Should show llama3.1:8b
```

### 2. FastAPI Backend

**Purpose:** Recommendation engine, workflow orchestration, API endpoints

**Start:**
```bash
make start-backend
```

**Manual start:**
```bash
cd backend
source venv/bin/activate
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

**Verify:**
```bash
curl http://localhost:8000/health
# Should return: {"status":"healthy"}
```

**API Documentation:**
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 3. Streamlit UI

**Purpose:** Conversational interface, recommendation display

**Start:**
```bash
make start-ui
```

**Manual start:**
```bash
source venv/bin/activate
streamlit run ui/app.py
```

**Access:**
- http://localhost:8501

**Note:** UI runs from project root to access `docs/` assets

### 4. KIND Cluster (Optional)

**Purpose:** Local Kubernetes for deployment testing

**Start:**
```bash
make cluster-start
```

**Manual start:**
```bash
scripts/kind-cluster.sh start
```

**Verify:**
```bash
kubectl cluster-info
kubectl get pods -A
make cluster-status
```

## Development Workflows

### Quick Development Cycle

**Start all services:**
```bash
make dev
```

**Make code changes, then:**
- **Backend changes:** Auto-reloads (uvicorn `--reload` flag)
- **UI changes:** Refresh browser (Streamlit auto-detects changes)
- **Data changes:** Restart backend to reload JSON files

**Stop services:**
```bash
make stop
```

### Working on Specific Components

**Backend only:**
```bash
make start-backend
make logs-backend  # Tail logs
```

**UI only (requires backend running):**
```bash
make start-ui
make logs-ui
```

**Test API endpoints:**
```bash
# Get recommendation
curl -X POST http://localhost:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"message": "I need a chatbot for 1000 users"}'
```

### Cluster Development

**Create cluster:**
```bash
make cluster-start
# Builds simulator, creates cluster, loads image
```

**Deploy from UI:**
1. Get recommendation
2. Generate YAML
3. Click "Deploy to Kubernetes"
4. Monitor in "Deployment Management" tab

**Manual deployment:**
```bash
# After generating YAML via UI
kubectl apply -f generated_configs/kserve-inferenceservice.yaml
kubectl get inferenceservices
kubectl get pods
```

**Clean up deployments:**
```bash
make clean-deployments  # Delete all InferenceServices
```

**Restart cluster:**
```bash
make cluster-restart  # Fresh cluster
```

## Testing

### Unit Tests

Test individual components without external dependencies:

```bash
make test-unit
```

**Run specific test:**
```bash
cd backend
source venv/bin/activate
pytest tests/test_model_recommender.py -v
```

### Integration Tests

Test components with Ollama integration:

```bash
make test-integration
```

Requires Ollama running with `llama3.1:8b` model.

### End-to-End Tests

Test full workflow including Kubernetes deployment:

```bash
make test-e2e
```

Requires:
- Ollama running
- KIND cluster running
- Backend and UI services

### Workflow Test

Test the complete recommendation workflow:

```bash
make test-workflow
```

This runs `backend/test_workflow.py` which tests all 3 demo scenarios.

### Watch Mode

Run tests continuously on file changes:

```bash
make test-watch
```

## Debugging

### View Logs

**Backend logs:**
```bash
make logs-backend
# Or manually:
tail -f .pids/backend.pid.log
```

**UI logs:**
```bash
make logs-ui
# Or manually:
tail -f .pids/ui.pid.log
```

**Kubernetes pod logs:**
```bash
kubectl logs -f <pod-name>
kubectl describe pod <pod-name>
```

### Check Service Health

```bash
make health
```

Checks:
- Backend: http://localhost:8000/health
- UI: http://localhost:8501
- Ollama: http://localhost:11434/api/tags

### Debug Intent Extraction

Test LLM client directly:

```bash
cd backend
source venv/bin/activate
python -c "
from src.llm.ollama_client import OllamaClient
from src.context_intent.extractor import IntentExtractor

client = OllamaClient()
extractor = IntentExtractor(client)

message = 'I need a chatbot for 5000 users with low latency'
intent = extractor.extract_intent(message)
print(intent)
"
```

### Debug Recommendations

Test recommendation engine:

```bash
cd backend
source venv/bin/activate
python -c "
from src.orchestration.workflow import RecommendationWorkflow

workflow = RecommendationWorkflow()
rec = workflow.generate_recommendation('I need a chatbot for 1000 users')
print(rec)
"
```

### Debug Cluster Deployments

**Check InferenceService status:**
```bash
kubectl get inferenceservices
kubectl describe inferenceservice <deployment-id>
```

**Check pod status:**
```bash
kubectl get pods
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

**Port-forward to service:**
```bash
kubectl port-forward svc/<deployment-id>-predictor 8080:80
curl http://localhost:8080/health
```

## Making Changes

### Adding a New Model

1. Add model to `data/model_catalog.json`:
```json
{
  "model_id": "new-model-id",
  "name": "New Model Name",
  "size_parameters": "7B",
  "context_length": 8192,
  "supported_tasks": ["chat", "instruction_following"],
  "recommended_for": ["chatbot"],
  "domain_specialization": ["general"]
}
```

2. Add benchmarks to `data/benchmarks.json`
3. Restart backend: `make restart`

### Adding a New Use Case Template

1. Add template to `data/slo_templates.json`:
```json
{
  "use_case": "new_use_case",
  "description": "Description",
  "prompt_tokens_mean": 200,
  "generation_tokens_mean": 150,
  "ttft_p90_target_ms": 250,
  "tpot_p90_target_ms": 60,
  "e2e_p95_target_ms": 3000
}
```

2. Update `backend/src/context_intent/extractor.py` USE_CASE_MAP
3. Restart backend

### Modifying the UI

UI code is in `ui/app.py`. Changes auto-reload in the browser.

**Key sections:**
- `render_chat_interface()` - Chat input/history
- `render_recommendation()` - Recommendation tabs
- `render_deployment_management_tab()` - Cluster management

### Modifying the Recommendation Algorithm

**Model scoring:** `backend/src/recommendation/model_recommender.py`
- `_score_model()` - Adjust scoring weights

**Capacity planning:** `backend/src/recommendation/capacity_planner.py`
- `plan_capacity()` - GPU sizing logic
- `_calculate_required_replicas()` - Scaling calculations

**Traffic profiling:** `backend/src/recommendation/traffic_profile.py`
- `generate_profile()` - Traffic estimation
- `generate_slo_targets()` - SLO target generation

### Code Quality

**Lint code:**
```bash
make lint
```

**Format code:**
```bash
make format
```

**Both use the shared project venv at root.**

## Simulator Development

### Building the Simulator

```bash
make build-simulator
```

Creates `vllm-simulator:latest` Docker image.

### Testing the Simulator Locally

```bash
docker run -p 8080:8080 \
  -e MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3 \
  -e GPU_TYPE=NVIDIA-L4 \
  -e TENSOR_PARALLEL_SIZE=1 \
  vllm-simulator:latest

# Test
curl -X POST http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello", "max_tokens": 10}'
```

### Pushing to Quay.io

```bash
make push-simulator
```

Auto-prompts for login if not authenticated.

## Clean Up

**Remove generated files:**
```bash
make clean
```

**Remove everything (including venvs):**
```bash
make clean-all
```

**Remove cluster:**
```bash
make cluster-stop
```

## Useful Commands

**See all available make targets:**
```bash
make help
```

**Show configuration:**
```bash
make info
```

**Open UI in browser:**
```bash
make open-ui
```

**Open API docs:**
```bash
make open-backend
```
