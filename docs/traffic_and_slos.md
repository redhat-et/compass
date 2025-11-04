# Mapping LLM Use Cases to Traffic Profiles and Experience-Driven SLOs

## Overview

This document defines a unified framework for mapping **LLM use cases** to both their corresponding **traffic profiles** and **Service Level Objectives (SLOs)**.

The purpose is to guide capacity planning, hardware selection, and cost optimization by distinguishing between what is **technically feasible** and what is **necessary for good user experience**.

In practice, this framework supports a workflow like the following:

1. Determine the user’s **use case**.  
2. Map that use case to its **traffic profile** (input/output token lengths).  
3. Apply the corresponding **SLO targets** for TTFT, ITL, and E2E latency.  
4. Combine with any user or system constraints (e.g., cost limits, available GPUs).  
5. Evaluate benchmark data to identify (GPU, model, configuration) combinations that meet those SLOs.  
6. If throughput exceeds a single GPU’s capacity, scale horizontally with multiple instances.

---

## 1. Traffic Profile Definitions

Traffic profiles describe the shape of LLM workloads in terms of **prompt (input) tokens** and **completion (output) tokens**.  
Different applications naturally cluster around characteristic token ratios.

| **Prompt Tokens** | **Output Tokens** | **Pattern Description** | **Typical Use Cases** | **Notes** |
|-------------------:|------------------:|--------------------------|------------------------|------------|
| **512** | **256** | Medium input, short output | Chatbot, interactive Q&A, short code completions | Most common interactive workload; strongly latency-sensitive. |
| **1024** | **1024** | Long input, long output | Content generation, translation, detailed code generation | Balanced workloads; typically less latency-sensitive but throughput-heavy. |
| **4096** | **512** | Very long input, short output | Summarization, document analysis, RAG Q&A over long context | Prefill-dominated workloads where TTFT is the primary bottleneck. |
| **10240** | **1536** | Extra-long input, medium output | Multi-document summarization, research or legal analysis | Edge workloads for long-context models; extremely memory- and bandwidth-intensive. |

### Rationale

- **Input length** drives **prefill cost** → directly impacts *Time to First Token (TTFT)*.  
- **Output length** drives **generation time** → affects *Inter-Token Latency (ITL)* and *End-to-End (E2E)* latency.  
- Use cases with the same traffic profiles may still differ in **SLO strictness** based on their **user experience expectations**.

---

## 2. Unified Mapping of Use Cases to Traffic Profiles and SLOs

The table below consolidates both traffic profiles and experience-driven SLOs into a single mapping.  
This ensures that each use case can be directly associated with both its computational pattern and its latency targets.

| **Use Case** | **Typical Task Description** | **Traffic Profile** <br>(Prompt → Output Tokens) | **Experience Class** | **TTFT (p95)** | **ITL (p95)** | **E2E (p95)** | **Rationale / Notes** |
|---------------|------------------------------|--------------------------------|----------------------|----------------|----------------|----------------|----------------|
| **Chatbot / Q&A** | Conversational assistants, customer support, knowledge search | **512 → 256** | **Conversational / Instant** | ≤150 ms | ≤25 ms/token | ≤7 s | Highly interactive; perceived responsiveness drives satisfaction. |
| **Code Completion / Copilot** | Inline IDE completions, command suggestions | **512 → 256** | **Instant (UX-critical)** | ≤100 ms | ≤20 ms/token | ≤5 s | Sub-200 ms first-token latency essential for fluid typing experience. |
| **Detailed Code Generation** | Multi-function or multi-file code synthesis | **1024 → 1024** | **Interactive** | ≤300 ms | ≤30 ms/token | ≤25 s | Users tolerate short delay for larger outputs; quality prioritized over immediacy. |
| **Translation / Paraphrasing** | Language conversion or rewriting | **1024 → 1024** | **Deferred** | ≤400 ms | ≤35 ms/token | ≤35 s | Non-interactive; a few-second delay acceptable. |
| **Content Generation / Writing Assistant** | Blog posts, marketing copy, creative writing | **1024 → 1024** | **Deferred / Batch** | ≤500 ms | ≤35 ms/token | ≤40 s | Emphasis on completeness and coherence over latency. |
| **Summarization (Short Doc)** | Summarize medium-length text (articles, emails) | **4096 → 512** | **Interactive / Deferred** | ≤600 ms | ≤30 ms/token | ≤15–20 s | Prefill-heavy; user can wait briefly for summary. |
| **Document Analysis / RAG Q&A** | Context retrieval, long-document reasoning | **4096 → 512** | **Interactive** | ≤600 ms | ≤30 ms/token | ≤18 s | Prefill cost dominates; responsiveness helps iterative Q&A. |
| **Long Document Summarization** | Summarize multi-page reports or papers | **10240 → 1536** | **Deferred** | ≤1 s | ≤40 ms/token | ≤60 s | User expects processing delay; prioritize throughput. |
| **Research / Legal / Multi-Document Q&A** | Analytical reasoning across large corpora | **10240 → 1536** | **Batch / Offline** | ≤2 s | ≤45 ms/token | ≤75–90 s | Asynchronous processing; cost and throughput optimized. |

---

## 3. Experience Classes and User Expectations

The experience class determines *how fast the interaction must feel* to meet user expectations, independent of hardware capabilities.  
This makes SLOs **experience-driven**, not merely performance-driven.

| **Experience Class** | **User Expectation** | **Example Applications** | **Latency Tolerance** |
|----------------------|----------------------|---------------------------|------------------------|
| **Instant (UX-Critical)** | Feels real-time; user notices any delay | Code completion, inline assistants | TTFT ≤150 ms; ITL ≤25 ms |
| **Conversational** | Feels natural; output streams smoothly | Chatbots, Q&A, support bots | TTFT ≤300 ms; ITL ≤30 ms |
| **Interactive** | Some waiting acceptable | RAG workflows, analysis bots | TTFT ≤500 ms; ITL ≤35 ms |
| **Deferred** | User expects delay (spinner acceptable) | Translation, summarization | TTFT ≤1 s; ITL ≤40 ms |
| **Batch / Offline** | Fully asynchronous; throughput prioritized | Research, document processing | TTFT ≤2 s; ITL ≤45 ms |

---

## 4. Practical Implications

### Hardware and Cost Optimization
Two workloads can have the **same traffic profile** but **different latency requirements**:
- **Latency-sensitive use cases** (chat, code completion) justify **higher-end GPUs** (A100, H100) with lower batching and tighter scheduling.
- **Throughput-oriented use cases** (summarization, translation) can use **mid-range GPUs** (L40S, A10) with larger batches and lower cost.

### Deployment Strategy
| **Experience Class** | **Hardware Tier** | **Batching Strategy** | **Priority Goal** |
|----------------------|------------------|----------------------|-------------------|
| Instant / Conversational | Premium GPU (A100/H100) | Small batches (≤4) | Low TTFT & ITL |
| Interactive | Balanced GPU (L40S, A10G) | Medium batches (8–16) | Balance latency & throughput |
| Deferred / Batch | Cost GPU (A10, T4) | Large batches (≥16) | Maximize throughput / cost |

### Throughput Handling
If the required throughput exceeds the capacity of a single GPU:
- **Replicate instances** of the same type.
- Scale horizontally to achieve desired QPS (queries per second).
- Maintain per-instance SLO compliance before aggregation.

---

## 5. Summary

- Traffic profiles define computational **load shape** (prefill vs. generation ratio).  
- SLOs define **experience expectations** — what latency is *necessary* for users to feel the system is performing well.  
- Identical traffic patterns may have **different SLOs** due to distinct UX requirements.  
- Separating “traffic profile” (what the workload *is*) from “experience class” (what the workload *needs*) enables precise capacity planning.  
- Hardware, batching, and scheduling strategies should be tuned to **meet the target SLOs** at the lowest feasible cost.  
