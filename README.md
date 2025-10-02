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
├── backend/                    # FastAPI backend
│   ├── src/
│   │   ├── context_intent/     # Component 2: Intent extraction
│   │   ├── recommendation/     # Component 3: Model/GPU recommendations
│   │   ├── simulation/         # Component 4: What-if analysis
│   │   ├── deployment/         # Component 5: YAML generation
│   │   ├── knowledge_base/     # Component 6: Data access layer
│   │   ├── orchestration/      # Component 8: Workflow coordination
│   │   ├── llm/                # Component 7: Ollama LLM client
│   │   └── api/                # FastAPI routes
│   └── requirements.txt
├── frontend/                   # Streamlit UI (Component 1)
│   ├── app.py
│   ├── components/
│   └── requirements.txt
├── data/                       # Synthetic data for POC
│   ├── benchmarks.json         # Model performance data
│   ├── slo_templates.json      # Use case SLO defaults
│   ├── model_catalog.json      # Available models
│   ├── sample_outcomes.json    # Mock deployment results
│   └── demo_scenarios.json     # 3 demo scenarios
├── generated_configs/          # Output directory for YAML files
└── tests/
```

## Prerequisites

- **Python 3.11+**
- **Ollama** installed and running
  ```bash
  brew install ollama
  ollama serve  # Start Ollama service
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
./run_api.sh

# Terminal 3 - Start Streamlit UI
./run_ui.sh
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
cd backend
source venv/bin/activate
uvicorn src.api.routes:app --host 0.0.0.0 --port 8000 --reload
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
- ⏳ **Sprint 4**: YAML generation, mock monitoring dashboard
- ⏳ **Sprint 5**: KIND cluster setup, KServe installation
- ⏳ **Sprint 6**: End-to-end deployment simulation

## Key Technologies

| Component | Technology |
|-----------|-----------|
| Backend | FastAPI, Pydantic |
| Frontend | Streamlit |
| LLM | Ollama (llama3.2:3b) |
| Data | JSON files (Phase 1), PostgreSQL (Phase 2+) |
| YAML Generation | Jinja2 templates |
| Deployment (Simulated) | KIND, KServe, vLLM |

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
python test_workflow.py

# Test FastAPI endpoints
python -m src.api.routes  # Start server
curl -X POST http://localhost:8000/api/v1/test
```

## Next Steps

Sprint 3 complete! Next up (Sprint 4):
1. YAML generation for KServe/vLLM deployments
2. Mock monitoring dashboard
3. Deployment templates with Jinja2
4. Configuration validation

## Contributing

This is a POC/design phase repository. See [CLAUDE.md](CLAUDE.md) for AI assistant guidance when making changes.

## License

[To be determined]
