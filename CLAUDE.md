# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains the architecture design for the **AI Pre-Deployment Assistant**, an open-source system that guides users from concept to production-ready LLM deployments through conversational AI and intelligent capacity planning.

**Key Principle**: This is a **Phase 1 POC implementation** based on the architecture design. The core functionality is complete and working end-to-end.

## Repository Structure

- **docs/ARCHITECTURE.md**: Comprehensive system architecture document
  - 9 core components with technology recommendations
  - Enhanced data schemas for SLO-driven deployment planning
  - Phase 1 (3-month) vs Phase 2+ implementation strategy
  - Knowledge Base schemas with 7 data collections

- **docs/architecture-diagram.md**: Visual architecture representations
  - Mermaid component diagrams
  - Sequence diagrams showing end-to-end flows
  - State machine for workflow orchestration
  - Entity-relationship diagrams for data models

- **backend/**: Python backend implementation
  - Component modules: context_intent, recommendation, knowledge_base, orchestration, api
  - LLM integration with Ollama client
  - FastAPI REST endpoints with CORS support
  - Pydantic schemas for type safety

- **ui/**: Streamlit UI
  - Chat interface for conversational requirement gathering
  - Multi-tab recommendation display (Overview, Specifications, Performance, Cost, Monitoring)
  - Editable specifications with review mode
  - Action buttons for YAML generation and deployment
  - Monitoring dashboard with cluster status, SLO compliance, and inference testing

- **data/**: Synthetic benchmark and catalog data for POC
  - benchmarks.json: 24 model+GPU combinations with vLLM performance data
  - model_catalog.json: 10 approved models with metadata
  - slo_templates.json: 7 use case templates
  - demo_scenarios.json: 3 test scenarios

## Architecture Key Concepts

### Problem Being Solved
Deploying LLMs in production is complex - users struggle to:
- Translate business needs into infrastructure choices (model, GPU type, SLO targets)
- Avoid trial-and-error that wastes time and money
- Understand resource requirements before committing to expensive GPU deployments

### Solution Approach
A **4-stage conversational flow**:
1. **Understand Business Context** - Extract intent via natural language
2. **Provide Tailored Recommendations** - Suggest model + GPU configurations
3. **Enable Interactive Exploration** - What-if scenario analysis
4. **One-Click Deployment** - Generate KServe/vLLM configs and deploy

### Core Innovation: SLO-Driven Capacity Planning
The system translates high-level user intent into technical specifications:
- **User says**: "I need a chatbot for 1000 users, low latency is critical"
- **System generates**:
  - Traffic profile (avg prompt: 150 tokens, gen: 200 tokens, peak QPS: 100)
  - SLO targets (TTFT p90: 200ms, TPOT p90: 50ms, E2E p95: 2000ms)
  - GPU capacity plan (e.g., "2x NVIDIA L4 GPUs, independent replicas")
  - Cost estimate ($800/month)

### The 10 Core Components

1. **Conversational Interface Layer** - Streamlit UI
2. **Context & Intent Engine** - Extract structured specs from conversation
   - Use case → SLO template mapping
   - Auto-generate traffic profiles
3. **Recommendation Engine** (3 sub-components):
   - Traffic Profile Generator
   - Model Recommendation Engine
   - Capacity Planning Engine
4. **Simulation & Exploration Layer** - What-if analysis, spec editing
5. **Deployment Automation Engine** - Generate YAML, deploy to K8s
6. **Knowledge Base** - Benchmarks, SLO templates, model catalog, outcomes
7. **LLM Backend** - Powers conversational AI (Ollama with llama3.1:8b)
8. **Orchestration & Workflow Engine** - Coordinate multi-step flows
9. **Inference Observability** - Monitor deployed models (TTFT, TPOT, GPU utilization)
10. **vLLM Simulator** - GPU-free development and testing

### Critical Data Collections (Knowledge Base)
- **Model Benchmarks**: TTFT/TPOT/throughput for (model, GPU, tensor_parallel) tuples
- **Use Case SLO Templates**: Default targets for chatbot, summarization, code-gen, etc.
- **Model Catalog**: Curated, approved models with task/domain metadata
- **Deployment Outcomes**: Actual performance data for feedback loop

## Working with This Repository

### When Modifying Architecture Documents

**docs/ARCHITECTURE.md and docs/architecture-diagram.md must stay synchronized**:
- If you change component descriptions in ARCHITECTURE.md, update architecture-diagram.md diagrams
- If you add/remove components, update both files
- Component numbering must match (e.g., "Component 3" in both docs)

### Key Architectural Decisions to Preserve

1. **Phase 1 uses Python** for all components (rapid development, stack consistency)
   - Go migration for Deployment Automation Engine is a possible future option (see Possible Future Enhancements in ARCHITECTURE.md)

2. **Phase 1 uses point estimates** for traffic (avg prompt length, avg QPS)
   - Benchmarks collected using vLLM default configuration (dynamic batching enabled)
   - Phase 2 adds full statistical distributions (mean, variance, tail) and multi-dimensional benchmarks

3. **SLO metrics are mandatory**:
   - TTFT (Time to First Token): p50, p90, p99 - **stored in benchmarks**
   - TPOT (Time Per Output Token): p50, p90, p99 - **stored in benchmarks**
   - E2E Latency: p50, p95, p99 - **calculated dynamically** from TTFT + (generation_tokens × TPOT)
   - Throughput: requests/sec and tokens/sec
   - Rationale: E2E latency varies by workload (generation length, streaming mode, use case), so it's calculated per-request rather than stored as a fixed benchmark value

4. **Editable specifications**: Users must be able to review and modify auto-generated specs before deployment

5. **Feedback loop**: Actual deployment outcomes feed back into Knowledge Base to improve future recommendations

### Terminology Consistency

- Use "**Pre-Deployment Assistant**" (not "Pre-Deployment Assistance")
- Use "**TTFT**" for Time to First Token (not "time-to-first-token")
- Use "**TPOT**" for Time Per Output Token
- Use "**SLO**" for Service Level Objective
- Use "**E2E**" for End-to-End latency
- GPU configurations: "2x NVIDIA L4" or "4x A100-80GB" (not "2 L4s")

### Common Editing Patterns

**Adding a new use case template**:
1. Add to Context & Intent Engine's USE_CASE_TEMPLATES in docs/ARCHITECTURE.md
2. Add corresponding entry to data/slo_templates.json
3. Update Knowledge Base → Use Case SLO Templates schema in docs/ARCHITECTURE.md
4. Update examples in Simulation Layer if relevant

**Adding a new SLO metric**:
1. Update DeploymentIntent schema in Context & Intent Engine (docs/ARCHITECTURE.md)
2. Update MODEL_BENCHMARKS schema in Knowledge Base (docs/ARCHITECTURE.md)
3. Update data/benchmarks.json with new metric fields
4. Update Inference Observability section
5. Update dashboard example if applicable
6. Update docs/architecture-diagram.md data model ERD

**Adding a new component**:
1. Add numbered section to docs/ARCHITECTURE.md (maintain sequential numbering)
2. Update "Architecture Components" count in Overview
3. Add to docs/architecture-diagram.md component diagram
4. Create corresponding backend/src/<component>/ directory
5. Update sequence diagram if component participates in main flow
6. Update Phase 1 technology choices table if relevant

## Open Questions and Future Work

See "Open Questions for Refinement" section in docs/ARCHITECTURE.md for:
- Multi-tenancy isolation
- Security validation of generated configs
- Conversational clarification flow (future phase)
- Model catalog sync strategy

## Git Workflow

This repository follows standard commit practices:
- Descriptive commit messages
- Include DCO Signed-off-by line (use `git commit -s`)
- Include Co-Authored-By: Claude <noreply@anthropic.com> (for code written with Claude's assistance)
- **DO NOT include** the "🤖 Generated with [Claude Code]" line in commit messages
- Push to `main` branch (no PR process currently)

**IMPORTANT**: Commit messages must follow this exact format:

```
Add YAML generation module

Implement DeploymentGenerator with Jinja2 templates for KServe,
vLLM, HPA, and ServiceMonitor configurations.

Co-Authored-By: Claude <noreply@anthropic.com>
Signed-off-by: Your Name <your.email@example.com>
```

**DO NOT use this format:**
```
Add YAML generation module

Implement DeploymentGenerator with Jinja2 templates for KServe,
vLLM, HPA, and ServiceMonitor configurations.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

## Important Notes

- **Current Implementation Status**:
  - ✅ Project structure with synthetic data and LLM client
  - ✅ Core recommendation engine (intent extraction, traffic profiling, model recommendation, capacity planning)
  - ✅ Orchestration workflow and FastAPI backend
  - ✅ Streamlit UI with chat interface, recommendation display, and editable specifications
  - ✅ YAML generation (KServe/vLLM/HPA/ServiceMonitor) and deployment automation
  - ✅ KIND cluster support with KServe installation
  - ✅ Kubernetes deployment automation and real cluster status monitoring
  - ✅ vLLM simulator for GPU-free development
  - ✅ Inference testing UI with end-to-end deployment validation
- The Knowledge Base schemas are critical - any implementation must support all 7 collections
- SLO-driven capacity planning is the core differentiator - don't simplify this away
- Use synthetic data in data/ directory for POC; production would use a database (e.g., PostgreSQL)
- Benchmarks use vLLM default configuration with dynamic batching (no fixed batch_size)

## Simulator Mode vs Real vLLM

The system now supports two deployment modes:

### Simulator Mode (Default for Development)
- **Purpose**: GPU-free development and testing on local machines
- **Location**: `simulator/` directory contains the vLLM simulator service
- **Docker Image**: `vllm-simulator:latest` (single image for all models)
- **Configuration**: Set `DeploymentGenerator(simulator_mode=True)` in `backend/src/api/routes.py`
- **Benefits**:
  - No GPU hardware required
  - Fast deployment (~10-15 seconds to Ready)
  - Predictable behavior for demos
  - Works on KIND (Kubernetes in Docker)
  - Uses actual benchmark data for realistic latency simulation

### Real vLLM Mode (Production)
- **Purpose**: Actual model inference with GPUs
- **Configuration**: Set `DeploymentGenerator(simulator_mode=False)` in `backend/src/api/routes.py`
- **Requirements**:
  - GPU-enabled Kubernetes cluster
  - NVIDIA GPU Operator installed
  - HuggingFace token secret for model downloads
  - Sufficient GPU resources (based on recommendations)
- **Behavior**:
  - Downloads actual models from HuggingFace
  - Real GPU inference
  - Production-grade performance

### When to Use Each Mode

**Use Simulator Mode for:**
- Local development and testing
- UI/UX iteration
- Workflow validation
- Demos and presentations
- CI/CD testing (no GPU required)

**Use Real vLLM Mode for:**
- Production deployments
- Performance benchmarking
- Model quality validation
- GPU utilization testing

### Technical Details

The deployment template (`backend/src/deployment/templates/kserve-inferenceservice.yaml.j2`) uses Jinja2 conditionals:
- `{% if simulator_mode %}` - Uses `vllm-simulator:latest`, no GPU resources, fast health checks
- `{% else %}` - Uses `vllm/vllm-openai:v0.6.2`, requests GPUs, longer health checks

Single codebase supports both modes - just toggle the flag!
