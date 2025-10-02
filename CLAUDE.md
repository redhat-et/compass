# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains the architecture design for the **Red Hat AI Pre-Deployment Assistant**, a system that guides users from concept to production-ready LLM deployments through conversational AI and intelligent capacity planning.

**Key Principle**: This is a **Phase 1 POC implementation** based on the architecture design. Sprints 1-3 are complete (foundation + core recommendation engine + UI), with subsequent sprints building out the full system.

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
  - Component modules stubbed out (context_intent, recommendation, etc.)
  - LLM integration with Ollama client
  - FastAPI for REST endpoints (Sprint 2+)

- **frontend/**: Streamlit UI (Sprint 3+)

- **data/**: Synthetic benchmark and catalog data for POC

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

### The 9 Core Components

1. **Conversational Interface Layer** - Chainlit/Streamlit UI
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
7. **LLM Backend** - Powers conversational AI
8. **Orchestration & Workflow Engine** - Coordinate multi-step flows
9. **Inference Observability** - Monitor deployed models (TTFT, TPOT, GPU utilization)

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
   - Phase 2+ may migrate Deployment Automation Engine to Go for K8s integration

2. **Phase 1 uses point estimates** for traffic (avg prompt length, avg QPS)
   - Benchmarks collected using vLLM default configuration (dynamic batching enabled)
   - Phase 2 adds full statistical distributions (mean, variance, tail) and multi-dimensional benchmarks

3. **SLO metrics are mandatory**:
   - TTFT (Time to First Token): p50, p90, p99
   - TPOT (Time Per Output Token): p50, p90, p99
   - E2E Latency: p50, p95, p99
   - Throughput: requests/sec and tokens/sec

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
- Include "🤖 Generated with Claude Code" footer
- Co-Author: Claude <noreply@anthropic.com>
- Push to `main` branch (no PR process currently)

## Important Notes

- **Sprint 1-3 Complete**:
  - Sprint 1: Project structure, synthetic data, and LLM client implemented
  - Sprint 2: Core recommendation engine (intent extraction, traffic profiling, model recommendation, capacity planning), orchestration workflow, FastAPI backend
  - Sprint 3: Streamlit UI with chat interface, recommendation display, and editable specifications
- When implementing subsequent sprints, preserve the architectural principles documented in docs/
- The Knowledge Base schemas are critical - any implementation must support all 7 collections
- SLO-driven capacity planning is the core differentiator - don't simplify this away
- Use synthetic data in data/ directory for POC; production would use PostgreSQL
- Benchmarks use vLLM default configuration with dynamic batching (no fixed batch_size)
