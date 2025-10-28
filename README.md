# <img src="docs/compass-logo.ico" alt="Compass" width="32" style="vertical-align: middle;"/> Compass

**Confidently navigate LLM deployments from concept to production.**

---

## Overview

The system addresses a critical challenge: **how do you translate business requirements into the right model and infrastructure choices without expensive trial-and-error?**

Compass guides you from concept to production LLM deployments through SLO-driven capacity planning. Conversationally define your requirements—Compass translates them into traffic profiles, performance targets, and cost constraints. Get intelligent model and GPU recommendations based on real benchmarks. Explore alternatives, compare tradeoffs, deploy with one click, and monitor actual performance—staying on course as your needs evolve.

The code in this repository is currently a **Phase 1 POC** that demonstrates SLO-driven LLM deployment planning.  It implements the Compass architecture and is intended to serve as a foundation for further development.  It is currently fully working with the caveat that most of the data (including the benchmarks) are currently synthetic.

### Key Features

- **🗣️ Conversational Requirements Gathering** - Describe your use case in natural language
- **📊 SLO-Driven Capacity Planning** - Translate business needs into technical specifications (traffic profiles, latency targets, cost constraints)
- **🎯 Intelligent Recommendations** - Get optimal model + GPU configurations backed by real benchmark data
- **🔍 What-If Analysis** - Explore alternatives and compare cost vs. latency tradeoffs
- **⚡ One-Click Deployment** - Generate production-ready KServe/vLLM YAML and deploy to Kubernetes
- **📈 Performance Monitoring** - Track actual deployment status and test inference in real-time
- **💻 GPU-Free Development** - vLLM simulator enables local testing without GPU hardware

### How It Works

1. **Extract Intent** - LLM-powered analysis converts your description into structured requirements
2. **Generate Traffic Profile** - Estimate prompt/generation tokens, QPS, and peak load patterns
3. **Set SLO Targets** - Auto-generate TTFT, TPOT, and E2E latency targets based on use case
4. **Recommend Model + GPU** - Score models against your requirements using benchmark data
5. **Plan Capacity** - Calculate required GPUs, tensor parallelism, and replica count
6. **Generate & Deploy** - Create validated Kubernetes YAML and deploy to local or production clusters
7. **Monitor & Validate** - Track deployment status and test inference endpoints

### Prerequisites

**Required before running `make setup`:**
- **macOS or Linux** (Windows via WSL2)
- **Docker Desktop** (must be running)

**Installed automatically by `make setup`:**
- **Python 3.11+**
- **Ollama** - `brew install ollama`
- **kubectl** - `brew install kubectl`
- **KIND** - `brew install kind`

## Quick Start

**Get up and running in 3 commands:**

```bash
make setup          # Install dependencies, pull Ollama model
make cluster-start  # Create local KIND cluster with vLLM simulator
make dev            # Start all services (Ollama + Backend + UI)
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

**Stop everything:**
```bash
make stop           # Stop services
make cluster-stop   # Delete cluster (optional)
```

### Using Compass

1. **Describe your use case** in the chat interface
   - Example: "I need a customer service chatbot for 5000 users with low latency"
2. **Review recommendations** - Model, GPU configuration, SLO predictions, costs
3. **Edit specifications** if needed (traffic, SLO targets, constraints)
4. **Generate deployment YAML** - Click "Generate Deployment YAML"
5. **Deploy to cluster** - Click "Deploy to Kubernetes"
6. **Monitor deployment** - Switch to "Deployment Management" tab to see status
7. **Test inference** - Send test prompts once deployment is Ready

## Demo Scenarios

The POC includes 3 pre-configured scenarios (see [data/demo_scenarios.json](data/demo_scenarios.json)):

1. **Customer Service Chatbot** - High volume (5000 users), strict latency (<500ms)
   - Expected: Llama 3.1 8B on 2x A100-80GB

2. **Code Generation Assistant** - Developer team (500 users), quality > speed
   - Expected: Llama 3.1 70B on 4x A100-80GB (tensor parallel)

3. **Document Summarization** - Batch processing (2000 users/day), cost-sensitive
   - Expected: Mistral 7B on 2x A10G

## Architecture Highlights

Compass implements an **8-component architecture** with:

- **Conversational Interface** (Streamlit) - Chat-based requirement gathering with interactive exploration
- **Context & Intent Engine** - LLM-powered extraction of deployment specs
- **Recommendation Engine** - Traffic profiling, model scoring, capacity planning
- **Deployment Automation** - YAML generation and Kubernetes deployment
- **Knowledge Base** - Benchmarks, SLO templates, model catalog
- **LLM Backend** - Ollama (llama3.1:8b) for conversational AI
- **Orchestration** - Multi-step workflow coordination
- **Inference Observability** - Real-time deployment monitoring

**Development Tools:**
- **vLLM Simulator** - GPU-free local development and testing

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for detailed system design.

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


## Development Commands

```bash
make help           # Show all available commands
make dev            # Start all services
make stop           # Stop all services
make restart        # Restart all services
make logs-backend   # Tail backend logs
make logs-ui        # Tail UI logs
make cluster-status # Check Kubernetes cluster status
make test           # Run all tests
make clean          # Remove generated files
```

## vLLM Simulator Mode

Compass includes a **GPU-free simulator** for local development:

- **No GPU required** - Run deployments on any laptop
- **OpenAI-compatible API** - `/v1/completions` and `/v1/chat/completions`
- **Realistic latency** - Uses benchmark data to simulate TTFT/TPOT
- **Fast deployment** - Pods become Ready in ~10-15 seconds

**Simulator Mode (default):**
```python
# In backend/src/api/routes.py
deployment_generator = DeploymentGenerator(simulator_mode=True)
```

**Production Mode (requires GPU cluster):**
```python
deployment_generator = DeploymentGenerator(simulator_mode=False)
```

See [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md#vllm-simulator-details) for details.

## Documentation

- **[Developer Guide](docs/DEVELOPER_GUIDE.md)** - Development workflows, testing, debugging
- **[Architecture](docs/ARCHITECTURE.md)** - Detailed system design and component specifications
- **[Architecture Diagrams](docs/architecture-diagram.md)** - Visual system representations
- **[Logging Guide](docs/LOGGING.md)** - Logging system and debugging
- **[Claude Code Guidance](CLAUDE.md)** - AI assistant instructions for contributors

## Future Enhancements

Potential improvements for Phase 2:

1. **Production-Grade Ingress** - External access with TLS, authentication, rate limiting
2. **Production GPU Validation** - End-to-end testing with real GPU clusters
3. **Feedback Loop** - Actual metrics → benchmark updates
4. **Statistical Traffic Models** - Distributions (not just point estimates)
5. **Multi-Dimensional Benchmarks** - Batch size, sequence length variations
6. **Security Hardening** - YAML validation, RBAC, network policies
7. **Multi-Tenancy** - Namespaces, resource quotas, isolation

## Contributing

We are early in development of this project, but contributions are welcome. 

See [CLAUDE.md](CLAUDE.md) for AI assistant guidance when making changes.

## License

This project is licensed under Apache License 2.0. See the [LICENSE file](LICENSE) for details.
