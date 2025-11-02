#!/usr/bin/env python3
"""
vLLM Simulator Service

A mock vLLM-compatible API server that simulates realistic LLM inference behavior
without requiring GPUs. Uses benchmark data to simulate realistic latency and
returns canned responses based on prompt patterns.

This allows:
- Development and testing without GPU resources
- Predictable demo behavior
- End-to-end workflow validation
- Fast iteration on deployment logic
"""

import os
import json
import time
import uuid
import random
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load configuration from environment
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", "1"))
GPU_TYPE = os.getenv("GPU_TYPE", "H100")
PORT = int(os.getenv("PORT", "8080"))

app = FastAPI(
    title="vLLM Simulator",
    description="Simulated vLLM OpenAI-compatible API",
    version="1.0.0"
)


# Request/Response models matching OpenAI API
class CompletionRequest(BaseModel):
    """Request for text completion"""
    prompt: str
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class ChatMessage(BaseModel):
    """Chat message"""
    role: str  # "system", "user", "assistant"
    content: str


class ChatCompletionRequest(BaseModel):
    """Request for chat completion"""
    messages: List[ChatMessage]
    max_tokens: Optional[int] = 100
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    stop: Optional[List[str]] = None


class BenchmarkLoader:
    """Load and query benchmark data"""

    def __init__(self, benchmarks_path: str = "/app/data/benchmarks.json"):
        self.benchmarks_path = benchmarks_path
        self.benchmarks = self._load_benchmarks()
        self.model_perf = self._find_benchmark()

    def _load_benchmarks(self) -> List[Dict[str, Any]]:
        """Load benchmark data from JSON file"""
        try:
            with open(self.benchmarks_path, 'r') as f:
                data = json.load(f)
                # Handle both formats: array at root or under "benchmarks" key
                if isinstance(data, list):
                    benchmarks = data
                elif isinstance(data, dict) and "benchmarks" in data:
                    benchmarks = data["benchmarks"]
                else:
                    logger.warning("Unexpected benchmark data format")
                    return []

                logger.info(f"Loaded {len(benchmarks)} benchmark entries")
                return benchmarks
        except FileNotFoundError:
            logger.warning(f"Benchmarks file not found: {self.benchmarks_path}")
            return []

    def _find_benchmark(self) -> Dict[str, float]:
        """Find benchmark data matching current model/GPU configuration"""
        # Extract model name from full ID (e.g., "mistralai/Mistral-7B-Instruct-v0.3" -> "Mistral 7B")
        model_short = self._normalize_model_name(MODEL_NAME)

        for bench in self.benchmarks:
            bench_model_id = bench.get("model_hf_repo", "")
            bench_model = self._normalize_model_name(bench_model_id)
            bench_gpu = bench.get("hardware", "")
            bench_tp = bench.get("hardware_count", 1)

            if (model_short in bench_model and
                GPU_TYPE in bench_gpu and
                bench_tp == TENSOR_PARALLEL_SIZE):
                logger.info(f"Found matching benchmark: {bench_model} on {bench_gpu} (TP={bench_tp})")
                return bench

        # Fallback to default values if no match
        logger.warning(f"No benchmark found for {MODEL_NAME} on {GPU_TYPE} (TP={TENSOR_PARALLEL_SIZE}), using defaults")
        return self._default_benchmark()

    def _normalize_model_name(self, name: str) -> str:
        """Normalize model name for comparison"""
        # Convert "mistralai/Mistral-7B-Instruct-v0.3" -> "mistral 7b"
        name = name.lower()
        name = name.split('/')[-1]  # Remove org prefix
        name = name.replace('-', ' ').replace('_', ' ')
        return name

    def _default_benchmark(self) -> Dict[str, float]:
        """Default benchmark values when no match found"""
        return {
            "model_hf_repo": MODEL_NAME,
            "hardware": GPU_TYPE,
            "hardware_count": TENSOR_PARALLEL_SIZE,
            "ttft_mean": 100.0,
            "ttft_p90": 150.0,
            "ttft_p95": 175.0,
            "ttft_p99": 200.0,
            "itl_mean": 30.0,
            "itl_p90": 45.0,
            "itl_p95": 52.5,
            "itl_p99": 60.0,
            "e2e_mean": 500.0,
            "e2e_p90": 750.0,
            "e2e_p95": 875.0,
            "e2e_p99": 1000.0,
            "tokens_per_second": 500.0,
            "requests_per_second": 10.0
        }

    def get_ttft(self, percentile: str = "mean") -> float:
        """Get TTFT for given percentile in seconds"""
        # Schema uses 'mean', 'p90', 'p95', 'p99' (values in ms)
        key = f"ttft_{percentile}"
        value_ms = self.model_perf.get(key, 100.0)
        return value_ms / 1000.0

    def get_itl(self, percentile: str = "mean") -> float:
        """Get ITL (Inter-Token Latency) for given percentile in seconds"""
        key = f"itl_{percentile}"
        value_ms = self.model_perf.get(key, 30.0)
        return value_ms / 1000.0


class CannedResponses:
    """Library of canned responses based on prompt patterns"""

    RESPONSES = {
        "code": [
            "def hello_world():\n    \"\"\"Print a greeting message.\"\"\"\n    print('Hello, World!')\n    return 'Hello, World!'",
            "function fibonacci(n) {\n    if (n <= 1) return n;\n    return fibonacci(n - 1) + fibonacci(n - 2);\n}",
            "public class Calculator {\n    public int add(int a, int b) {\n        return a + b;\n    }\n}",
            "def binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1",
        ],
        "chat": [
            "I'd be happy to help you with that! What specific aspect would you like to know more about?",
            "That's a great question. Based on my understanding, I would recommend considering the following factors...",
            "I understand your concern. Let me break this down into simpler terms for you.",
            "Absolutely! Here's what you need to know about this topic.",
            "I appreciate you asking. The answer depends on several factors, but generally speaking...",
        ],
        "summarization": [
            "In summary: The key points are (1) efficient resource allocation, (2) cost optimization, and (3) performance monitoring.",
            "TL;DR: The main idea is to balance latency requirements with cost constraints while maintaining high availability.",
            "To summarize: This approach prioritizes throughput over latency for batch processing workloads.",
            "Executive summary: The proposed solution reduces infrastructure costs by 40% while maintaining SLO compliance.",
        ],
        "qa": [
            "Based on the context provided, the answer is: The system uses a recommendation engine to match business requirements with optimal infrastructure configurations.",
            "The correct answer is: You should choose GPU type based on your latency requirements and budget constraints.",
            "According to best practices: Start with the smallest GPU that meets your SLO targets, then scale horizontally if needed.",
        ],
        "creative": [
            "Once upon a time, in a world where machines learned from data, there was an AI assistant that helped users deploy their models efficiently.",
            "The algorithm pondered its existence, processing patterns in the vast sea of data, seeking optimal solutions to complex problems.",
            "In the realm of cloud computing, where resources were measured in milliseconds and megabytes, a new paradigm emerged.",
        ]
    }

    @classmethod
    def get_response(cls, prompt: str) -> str:
        """Get appropriate canned response based on prompt pattern"""
        prompt_lower = prompt.lower()

        # Pattern matching
        if any(kw in prompt_lower for kw in ["code", "function", "def ", "class ", "implement", "write a"]):
            category = "code"
        elif any(kw in prompt_lower for kw in ["summarize", "summary", "tldr", "in brief"]):
            category = "summarization"
        elif any(kw in prompt_lower for kw in ["what is", "explain", "how does", "why", "when"]):
            category = "qa"
        elif any(kw in prompt_lower for kw in ["story", "poem", "creative", "imagine", "write about"]):
            category = "creative"
        else:
            category = "chat"

        return random.choice(cls.RESPONSES[category])


# Initialize benchmark loader
benchmark_loader = BenchmarkLoader()

logger.info(f"vLLM Simulator initialized:")
logger.info(f"  Model: {MODEL_NAME}")
logger.info(f"  GPU Type: {GPU_TYPE}")
logger.info(f"  Tensor Parallel Size: {TENSOR_PARALLEL_SIZE}")
logger.info(f"  TTFT mean: {benchmark_loader.get_ttft('mean')*1000:.1f}ms")
logger.info(f"  ITL mean: {benchmark_loader.get_itl('mean')*1000:.1f}ms")


@app.get("/")
def root():
    """Root endpoint"""
    return {
        "message": "vLLM Simulator API",
        "model": MODEL_NAME,
        "simulator": True
    }


@app.get("/health")
def health():
    """Health check endpoint (matches vLLM)"""
    return {"status": "healthy"}


@app.get("/v1/models")
def list_models():
    """List available models (matches OpenAI API)"""
    return {
        "object": "list",
        "data": [
            {
                "id": MODEL_NAME,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "simulator",
                "permission": [],
                "root": MODEL_NAME,
                "parent": None
            }
        ]
    }


@app.post("/v1/completions")
def create_completion(request: CompletionRequest):
    """Create text completion (matches OpenAI API)"""
    logger.info(f"Completion request: prompt_length={len(request.prompt)}, max_tokens={request.max_tokens}")

    # Simulate TTFT (Time to First Token) - use mean for typical performance
    ttft = benchmark_loader.get_ttft("mean")
    time.sleep(ttft)

    # Get canned response
    response_text = CannedResponses.get_response(request.prompt)

    # Simulate ITL (Inter-Token Latency) for remaining tokens
    # Approximate tokens in response
    estimated_tokens = len(response_text.split())
    itl = benchmark_loader.get_itl("mean")
    time.sleep(itl * min(estimated_tokens, request.max_tokens))

    # Build response matching OpenAI format
    return {
        "id": f"cmpl-{uuid.uuid4()}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "text": response_text,
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(request.prompt.split()),
            "completion_tokens": estimated_tokens,
            "total_tokens": len(request.prompt.split()) + estimated_tokens
        }
    }


@app.post("/v1/chat/completions")
def create_chat_completion(request: ChatCompletionRequest):
    """Create chat completion (matches OpenAI API)"""
    # Extract last user message
    user_messages = [m for m in request.messages if m.role == "user"]
    last_message = user_messages[-1].content if user_messages else "Hello"

    logger.info(f"Chat completion request: messages={len(request.messages)}, max_tokens={request.max_tokens}")

    # Simulate TTFT - use mean for typical performance
    ttft = benchmark_loader.get_ttft("mean")
    time.sleep(ttft)

    # Get canned response
    response_text = CannedResponses.get_response(last_message)

    # Simulate ITL (Inter-Token Latency)
    estimated_tokens = len(response_text.split())
    itl = benchmark_loader.get_itl("mean")
    time.sleep(itl * min(estimated_tokens, request.max_tokens))

    return {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": sum(len(m.content.split()) for m in request.messages),
            "completion_tokens": estimated_tokens,
            "total_tokens": sum(len(m.content.split()) for m in request.messages) + estimated_tokens
        }
    }


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint (matches vLLM format)"""
    # Simulate realistic metrics based on benchmark data
    perf = benchmark_loader.model_perf

    # Counter for total requests (simulated)
    total_requests = 1000

    # Get metrics using new field names
    ttft_mean = perf.get('ttft_mean', 100.0)
    itl_mean = perf.get('itl_mean', 30.0)

    metrics_text = f"""# HELP vllm_time_to_first_token_seconds Time to first token
# TYPE vllm_time_to_first_token_seconds histogram
vllm_time_to_first_token_seconds_sum {ttft_mean * total_requests / 1000}
vllm_time_to_first_token_seconds_count {total_requests}

# HELP vllm_time_per_output_token_seconds Time per output token
# TYPE vllm_time_per_output_token_seconds histogram
vllm_time_per_output_token_seconds_sum {itl_mean * total_requests / 1000}
vllm_time_per_output_token_seconds_count {total_requests}

# HELP vllm_request_duration_seconds Request duration
# TYPE vllm_request_duration_seconds histogram
vllm_request_duration_seconds_sum {(ttft_mean + itl_mean * 50) * total_requests / 1000}
vllm_request_duration_seconds_count {total_requests}

# HELP vllm_prompt_tokens_total Total prompt tokens processed
# TYPE vllm_prompt_tokens_total counter
vllm_prompt_tokens_total {total_requests * 150}

# HELP vllm_generation_tokens_total Total generation tokens produced
# TYPE vllm_generation_tokens_total counter
vllm_generation_tokens_total {total_requests * 200}

# HELP vllm_request_success_total Total successful requests
# TYPE vllm_request_success_total counter
vllm_request_success_total {total_requests}

# HELP vllm_gpu_cache_usage_perc GPU KV cache usage percentage
# TYPE vllm_gpu_cache_usage_perc gauge
vllm_gpu_cache_usage_perc 45.2

# HELP vllm_num_requests_running Number of requests currently running
# TYPE vllm_num_requests_running gauge
vllm_num_requests_running 3

# HELP vllm_num_requests_waiting Number of requests waiting in queue
# TYPE vllm_num_requests_waiting gauge
vllm_num_requests_waiting 0
"""

    return metrics_text


if __name__ == "__main__":
    logger.info(f"Starting vLLM Simulator on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
