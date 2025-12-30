#!/usr/bin/env python3
"""
Generate estimated performance data for high-accuracy models without benchmarks.

Approach:
1. Identify top 20 high-accuracy models without performance data
2. For each model, assign best use case based on accuracy scores
3. Use token config for that use case
4. Estimate performance within known benchmark ranges
5. Output to separate JSON, then merge with main benchmarks
"""

import json
import os
from pathlib import Path
from datetime import datetime

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
PERFORMANCE_JSON = DATA_DIR / "benchmarks_redhat_performance.json"
OUTPUT_JSON = DATA_DIR / "benchmarks_estimated_performance.json"

# Token configs by use case
TOKEN_CONFIGS = {
    "chatbot_conversational": {"prompt": 512, "output": 256},
    "code_completion": {"prompt": 512, "output": 256},
    "code_generation_detailed": {"prompt": 1024, "output": 1024},
    "summarization_short": {"prompt": 4096, "output": 512},
    "research_legal_analysis": {"prompt": 10240, "output": 1536},
}

# Hardware by model size (in billions of parameters)
# IMPORTANT: Same GPU count for comparable hardware tiers to ensure H200 > H100 > A100
def get_hardware_config(size_b: float, is_quantized: bool = False):
    """Get appropriate hardware based on model size.
    
    Key principle: For same GPU count, better hardware = faster performance.
    """
    if size_b <= 8:
        return [
            {"hardware": "L4", "count": 1},
            {"hardware": "A100-80", "count": 1},
            {"hardware": "H100", "count": 1},
        ]
    elif size_b <= 24:
        return [
            {"hardware": "A100-80", "count": 1},
            {"hardware": "H100", "count": 1},
            {"hardware": "H200", "count": 1},
        ]
    elif size_b <= 70:
        return [
            {"hardware": "H100", "count": 2},
            {"hardware": "H200", "count": 2},  # Same count for fair comparison
        ]
    elif size_b <= 200:
        return [
            {"hardware": "H200", "count": 4},
            {"hardware": "B200", "count": 4},  # Same count for fair comparison
        ]
    else:  # >200B
        return [
            {"hardware": "H200", "count": 8},
            {"hardware": "B200", "count": 8},  # Same count for fair comparison
        ]


# Reference performance patterns from existing benchmarks (averaged)
# Format: {size_category: {metric: value}}
REFERENCE_PERFORMANCE = {
    "tiny": {  # ≤3B
        "ttft_mean": 25, "ttft_p90": 35, "ttft_p95": 42, "ttft_p99": 55,
        "itl_mean": 8, "itl_p90": 12, "itl_p95": 15, "itl_p99": 20,
        "e2e_mean": 2500, "e2e_p90": 3200, "e2e_p95": 3800, "e2e_p99": 4500,
        "tps_mean": 450, "rps": 4.0,
    },
    "small": {  # 3-10B
        "ttft_mean": 45, "ttft_p90": 65, "ttft_p95": 78, "ttft_p99": 95,
        "itl_mean": 12, "itl_p90": 18, "itl_p95": 22, "itl_p99": 28,
        "e2e_mean": 3500, "e2e_p90": 4500, "e2e_p95": 5200, "e2e_p99": 6200,
        "tps_mean": 320, "rps": 3.0,
    },
    "medium": {  # 10-30B
        "ttft_mean": 75, "ttft_p90": 110, "ttft_p95": 130, "ttft_p99": 160,
        "itl_mean": 18, "itl_p90": 26, "itl_p95": 32, "itl_p99": 42,
        "e2e_mean": 5500, "e2e_p90": 7000, "e2e_p95": 8200, "e2e_p99": 10000,
        "tps_mean": 220, "rps": 2.0,
    },
    "large": {  # 30-80B
        "ttft_mean": 120, "ttft_p90": 175, "ttft_p95": 210, "ttft_p99": 260,
        "itl_mean": 25, "itl_p90": 38, "itl_p95": 46, "itl_p99": 58,
        "e2e_mean": 8000, "e2e_p90": 10500, "e2e_p95": 12500, "e2e_p99": 15000,
        "tps_mean": 150, "rps": 1.5,
    },
    "xlarge": {  # 80-250B
        "ttft_mean": 200, "ttft_p90": 290, "ttft_p95": 350, "ttft_p99": 430,
        "itl_mean": 35, "itl_p90": 52, "itl_p95": 65, "itl_p99": 82,
        "e2e_mean": 12000, "e2e_p90": 16000, "e2e_p95": 19000, "e2e_p99": 23000,
        "tps_mean": 100, "rps": 1.0,
    },
    "huge": {  # >250B
        "ttft_mean": 350, "ttft_p90": 500, "ttft_p95": 600, "ttft_p99": 750,
        "itl_mean": 50, "itl_p90": 75, "itl_p95": 92, "itl_p99": 115,
        "e2e_mean": 20000, "e2e_p90": 26000, "e2e_p95": 31000, "e2e_p99": 38000,
        "tps_mean": 65, "rps": 0.5,
    },
}


def get_size_category(size_b: float) -> str:
    """Get size category for a model."""
    if size_b <= 3:
        return "tiny"
    elif size_b <= 10:
        return "small"
    elif size_b <= 30:
        return "medium"
    elif size_b <= 80:
        return "large"
    elif size_b <= 250:
        return "xlarge"
    else:
        return "huge"


def scale_for_token_config(base_perf: dict, prompt_tokens: int, output_tokens: int) -> dict:
    """Scale performance based on token configuration."""
    # Base reference is 512/256
    base_prompt, base_output = 512, 256
    
    # TTFT scales with prompt tokens (prefill)
    ttft_factor = (prompt_tokens / base_prompt) ** 0.7  # Sub-linear scaling
    
    # ITL is relatively stable (generation phase)
    itl_factor = 1.0 + (output_tokens - base_output) / base_output * 0.1
    
    # E2E scales with total tokens
    total_factor = ((prompt_tokens + output_tokens) / (base_prompt + base_output)) ** 0.6
    
    # TPS inversely scales with complexity
    tps_factor = 1.0 / (1.0 + (prompt_tokens - base_prompt) / base_prompt * 0.3)
    
    return {
        "ttft_mean": int(base_perf["ttft_mean"] * ttft_factor),
        "ttft_p90": int(base_perf["ttft_p90"] * ttft_factor),
        "ttft_p95": int(base_perf["ttft_p95"] * ttft_factor),
        "ttft_p99": int(base_perf["ttft_p99"] * ttft_factor),
        "itl_mean": int(base_perf["itl_mean"] * itl_factor),
        "itl_p90": int(base_perf["itl_p90"] * itl_factor),
        "itl_p95": int(base_perf["itl_p95"] * itl_factor),
        "itl_p99": int(base_perf["itl_p99"] * itl_factor),
        "e2e_mean": int(base_perf["e2e_mean"] * total_factor),
        "e2e_p90": int(base_perf["e2e_p90"] * total_factor),
        "e2e_p95": int(base_perf["e2e_p95"] * total_factor),
        "e2e_p99": int(base_perf["e2e_p99"] * total_factor),
        "tps_mean": int(base_perf["tps_mean"] * tps_factor),
        "rps": round(base_perf["rps"] * tps_factor, 1),
    }


def scale_for_hardware(base_perf: dict, hardware: str) -> dict:
    """Adjust performance based on hardware tier."""
    # Hardware performance factors (relative to H100)
    hw_factors = {
        "L4": {"speed": 0.4, "throughput": 0.3},      # Entry level
        "A100-40": {"speed": 0.7, "throughput": 0.6},  # Mid tier
        "A100-80": {"speed": 0.85, "throughput": 0.75}, # Good
        "H100": {"speed": 1.0, "throughput": 1.0},     # Reference
        "H200": {"speed": 1.3, "throughput": 1.4},     # High end
        "B200": {"speed": 1.5, "throughput": 1.6},     # Top tier
    }
    
    factor = hw_factors.get(hardware, {"speed": 1.0, "throughput": 1.0})
    speed = factor["speed"]
    throughput = factor["throughput"]
    
    return {
        "ttft_mean": int(base_perf["ttft_mean"] / speed),
        "ttft_p90": int(base_perf["ttft_p90"] / speed),
        "ttft_p95": int(base_perf["ttft_p95"] / speed),
        "ttft_p99": int(base_perf["ttft_p99"] / speed),
        "itl_mean": int(base_perf["itl_mean"] / speed),
        "itl_p90": int(base_perf["itl_p90"] / speed),
        "itl_p95": int(base_perf["itl_p95"] / speed),
        "itl_p99": int(base_perf["itl_p99"] / speed),
        "e2e_mean": int(base_perf["e2e_mean"] / speed),
        "e2e_p90": int(base_perf["e2e_p90"] / speed),
        "e2e_p95": int(base_perf["e2e_p95"] / speed),
        "e2e_p99": int(base_perf["e2e_p99"] / speed),
        "tps_mean": int(base_perf["tps_mean"] * throughput),
        "rps": round(base_perf["rps"] * throughput, 1),
    }


# Top 20 models to estimate (high accuracy, no performance data)
MODELS_TO_ESTIMATE = [
    {
        "model_id": "deepseek/DeepSeek-V3.2-Exp-Reasoning",
        "model_name": "DeepSeek V3.2 Exp (Reasoning)",
        "size_b": 670,
        "best_use_case": "code_generation_detailed",
        "based_on": "DeepSeek R1 family patterns",
    },
    {
        "model_id": "deepseek/DeepSeek-V3.1-Terminus-Reasoning",
        "model_name": "DeepSeek V3.1 Terminus (Reasoning)",
        "size_b": 670,
        "best_use_case": "code_generation_detailed",
        "based_on": "DeepSeek R1 family patterns",
    },
    {
        "model_id": "minimax/MiniMax-M2",
        "model_name": "MiniMax-M2",
        "size_b": 456,
        "best_use_case": "research_legal_analysis",
        "based_on": "Large model patterns",
    },
    {
        "model_id": "mistralai/Magistral-Medium-1.2",
        "model_name": "Magistral Medium 1.2",
        "size_b": 123,
        "best_use_case": "code_generation_detailed",
        "based_on": "Mistral family patterns",
    },
    {
        "model_id": "mistralai/Magistral-Small-1.2",
        "model_name": "Magistral Small 1.2",
        "size_b": 24,
        "best_use_case": "code_completion",
        "based_on": "Mistral Small 3.1 patterns",
    },
    {
        "model_id": "nvidia/Llama-Nemotron-Super-49B-v1.5-Reasoning",
        "model_name": "Llama Nemotron Super 49B v1.5 (Reasoning)",
        "size_b": 49,
        "best_use_case": "code_generation_detailed",
        "based_on": "Nemotron 70B patterns",
    },
    {
        "model_id": "upstage/Solar-Pro-2-Reasoning",
        "model_name": "Solar Pro 2 (Reasoning)",
        "size_b": 22,
        "best_use_case": "code_generation_detailed",
        "based_on": "Medium model patterns",
    },
    {
        "model_id": "deepseek/DeepSeek-R1-Distill-Llama-70B",
        "model_name": "DeepSeek R1 Distill Llama 70B",
        "size_b": 70,
        "best_use_case": "chatbot_conversational",
        "based_on": "Llama 70B patterns",
    },
    {
        "model_id": "google/Gemma-3-27B-Instruct",
        "model_name": "Gemma 3 27B Instruct",
        "size_b": 27,
        "best_use_case": "chatbot_conversational",
        "based_on": "Gemma family patterns",
    },
    {
        "model_id": "google/Gemma-3-12B-Instruct",
        "model_name": "Gemma 3 12B Instruct",
        "size_b": 12,
        "best_use_case": "chatbot_conversational",
        "based_on": "Gemma family patterns",
    },
    {
        "model_id": "meta-llama/Llama-3.1-405B-Instruct",
        "model_name": "Llama 3.1 Instruct 405B",
        "size_b": 405,
        "best_use_case": "summarization_short",
        "based_on": "Llama family patterns",
    },
    {
        "model_id": "mistralai/Mistral-Medium-3.1",
        "model_name": "Mistral Medium 3.1",
        "size_b": 123,
        "best_use_case": "chatbot_conversational",
        "based_on": "Mistral family patterns",
    },
    {
        "model_id": "mistralai/Mistral-Small-3.2",
        "model_name": "Mistral Small 3.2",
        "size_b": 24,
        "best_use_case": "chatbot_conversational",
        "based_on": "Mistral Small 3.1 patterns",
    },
    {
        "model_id": "deepseek/DeepSeek-R1-0528-Qwen3-8B",
        "model_name": "DeepSeek R1 0528 Qwen3 8B",
        "size_b": 8,
        "best_use_case": "code_completion",
        "based_on": "Qwen3-8B patterns",
    },
    {
        "model_id": "qwen/Qwen3-235B-A22B-Thinking",
        "model_name": "Qwen3 235B A22B (Thinking)",
        "size_b": 235,
        "best_use_case": "research_legal_analysis",
        "based_on": "Qwen family patterns",
    },
    {
        "model_id": "microsoft/Phi-4-Mini-Instruct",
        "model_name": "Phi-4 Mini Instruct",
        "size_b": 3.8,
        "best_use_case": "chatbot_conversational",
        "based_on": "Phi-4 patterns",
    },
    {
        "model_id": "nvidia/Nemotron-Nano-9B-V2-Reasoning",
        "model_name": "NVIDIA Nemotron Nano 9B V2 (Reasoning)",
        "size_b": 9,
        "best_use_case": "code_completion",
        "based_on": "Nemotron Nano patterns",
    },
    {
        "model_id": "nvidia/Llama-3.3-Nemotron-Super-49B-v1-Reasoning",
        "model_name": "Llama 3.3 Nemotron Super 49B v1 (Reasoning)",
        "size_b": 49,
        "best_use_case": "code_generation_detailed",
        "based_on": "Nemotron patterns",
    },
    {
        "model_id": "upstage/Solar-Pro-2-Non-reasoning",
        "model_name": "Solar Pro 2 (Non-reasoning)",
        "size_b": 22,
        "best_use_case": "chatbot_conversational",
        "based_on": "Medium model patterns",
    },
    {
        "model_id": "deepseek/DeepSeek-V3.2-Exp-Non-reasoning",
        "model_name": "DeepSeek V3.2 Exp (Non-reasoning)",
        "size_b": 670,
        "best_use_case": "summarization_short",
        "based_on": "DeepSeek patterns",
    },
]


def generate_benchmark_config(model: dict, hw_config: dict, token_config: dict) -> dict:
    """Generate a single benchmark configuration for a model."""
    size_cat = get_size_category(model["size_b"])
    base_perf = REFERENCE_PERFORMANCE[size_cat].copy()
    
    # Scale for token config
    scaled_perf = scale_for_token_config(
        base_perf, 
        token_config["prompt"], 
        token_config["output"]
    )
    
    # Scale for hardware
    final_perf = scale_for_hardware(scaled_perf, hw_config["hardware"])
    
    # Scale for GPU count (more GPUs = faster, but not linear)
    gpu_factor = 1.0 / (hw_config["count"] ** 0.5)  # Square root scaling
    for key in ["ttft_mean", "ttft_p90", "ttft_p95", "ttft_p99", 
                "itl_mean", "itl_p90", "itl_p95", "itl_p99",
                "e2e_mean", "e2e_p90", "e2e_p95", "e2e_p99"]:
        final_perf[key] = int(final_perf[key] * gpu_factor)
    
    # TPS and RPS scale up with more GPUs
    final_perf["tps_mean"] = int(final_perf["tps_mean"] * (hw_config["count"] ** 0.7))
    final_perf["rps"] = round(final_perf["rps"] * (hw_config["count"] ** 0.6), 1)
    
    return {
        "model_id": model["model_id"],
        "model_name": model["model_name"],
        "hardware": hw_config["hardware"],
        "hardware_count": hw_config["count"],
        "prompt_tokens": token_config["prompt"],
        "output_tokens": token_config["output"],
        "ttft_mean": final_perf["ttft_mean"],
        "ttft_p90": final_perf["ttft_p90"],
        "ttft_p95": final_perf["ttft_p95"],
        "ttft_p99": final_perf["ttft_p99"],
        "itl_mean": final_perf["itl_mean"],
        "itl_p90": final_perf["itl_p90"],
        "itl_p95": final_perf["itl_p95"],
        "itl_p99": final_perf["itl_p99"],
        "e2e_mean": final_perf["e2e_mean"],
        "e2e_p90": final_perf["e2e_p90"],
        "e2e_p95": final_perf["e2e_p95"],
        "e2e_p99": final_perf["e2e_p99"],
        "tokens_per_second_mean": final_perf["tps_mean"],
        "requests_per_second": final_perf["rps"],
        "estimated": True,
        "estimation_method": "family_size_interpolation",
        "estimation_confidence": 0.75,
        "based_on": model["based_on"],
        "best_use_case": model["best_use_case"],
    }


def main():
    print("=" * 60)
    print("Generating Estimated Performance Data")
    print("=" * 60)
    
    all_benchmarks = []
    
    for model in MODELS_TO_ESTIMATE:
        print(f"\nProcessing: {model['model_name']} ({model['size_b']}B)")
        
        # Get token config for best use case
        token_config = TOKEN_CONFIGS[model["best_use_case"]]
        print(f"  Use case: {model['best_use_case']} → Tokens: {token_config['prompt']}/{token_config['output']}")
        
        # Get hardware configurations
        hw_configs = get_hardware_config(model["size_b"])
        
        for hw in hw_configs:
            config = generate_benchmark_config(model, hw, token_config)
            all_benchmarks.append(config)
            print(f"  → {hw['hardware']} x{hw['count']}: TTFT={config['ttft_p95']}ms, ITL={config['itl_p95']}ms, TPS={config['tokens_per_second_mean']}")
    
    # Create output structure
    output = {
        "metadata": {
            "source": "estimated_performance_interpolation",
            "description": "Estimated performance data for high-accuracy models without real benchmarks",
            "generated_at": datetime.now().isoformat(),
            "total_configs": len(all_benchmarks),
            "unique_models": len(MODELS_TO_ESTIMATE),
            "estimation_method": "family_size_interpolation with hardware scaling",
            "confidence_level": 0.75,
            "disclaimer": "These are ESTIMATED values based on similar models. Real benchmarks may differ.",
            "models": [m["model_name"] for m in MODELS_TO_ESTIMATE],
        },
        "benchmarks": all_benchmarks,
    }
    
    # Save to separate file
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"Generated {len(all_benchmarks)} benchmark configurations")
    print(f"For {len(MODELS_TO_ESTIMATE)} models")
    print(f"Saved to: {OUTPUT_JSON}")
    print(f"{'=' * 60}")
    
    return output


def merge_with_main_benchmarks():
    """Merge estimated benchmarks with main performance file."""
    print("\nMerging with main benchmarks...")
    
    # Load estimated data
    with open(OUTPUT_JSON, 'r') as f:
        estimated = json.load(f)
    
    # Load main benchmarks
    with open(PERFORMANCE_JSON, 'r') as f:
        main_data = json.load(f)
    
    # Add estimated benchmarks
    main_data["benchmarks"].extend(estimated["benchmarks"])
    
    # Update metadata
    main_data["metadata"]["total_configs"] = len(main_data["benchmarks"])
    main_data["metadata"]["unique_models"] += estimated["metadata"]["unique_models"]
    main_data["metadata"]["models"].extend(estimated["metadata"]["models"])
    main_data["metadata"]["includes_estimated"] = True
    main_data["metadata"]["estimated_models_count"] = estimated["metadata"]["unique_models"]
    
    # Save merged file
    with open(PERFORMANCE_JSON, 'w') as f:
        json.dump(main_data, f, indent=2)
    
    print(f"Merged! Total configs: {main_data['metadata']['total_configs']}")
    print(f"Total models: {main_data['metadata']['unique_models']}")


if __name__ == "__main__":
    output = main()
    
    # Ask user if they want to merge
    print("\nDo you want to merge estimated data with main benchmarks? (y/n)")
    # For script execution, we'll do it automatically
    # merge_with_main_benchmarks()

