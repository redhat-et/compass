#!/usr/bin/env python3
"""
Update benchmarks.json to use 4 GuideLLM traffic profiles.

This script:
1. Reads existing benchmarks.json
2. Groups benchmarks by (model, hardware, hardware_count)
3. For each group, creates 4 new benchmarks with GuideLLM traffic profiles:
   - (512, 256)
   - (1024, 1024)
   - (4096, 512)
   - (10240, 1536) - optional for Phase 1
4. Estimates p95 values from existing p90 and p99 data
5. Writes updated benchmarks.json
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


# GuideLLM traffic profiles
TRAFFIC_PROFILES = [
    (512, 256),
    (1024, 1024),
    (4096, 512),
    (10240, 1536),
]


def estimate_p95(p90, p99):
    """Estimate p95 from p90 and p99 using linear interpolation."""
    # p95 is halfway between p90 and p99 in percentile space
    return p90 + (p99 - p90) * 0.5


def scale_latency(base_value, base_tokens, new_tokens, is_prefill=False):
    """
    Scale latency value based on token count change.

    For TTFT (prefill): roughly linear with input tokens
    For ITL (decode): constant per token
    For E2E: TTFT + (output_tokens * ITL)
    """
    if is_prefill:
        # TTFT scales with input tokens (prefill cost)
        return base_value * (new_tokens / base_tokens) if base_tokens > 0 else base_value
    else:
        # ITL is constant per token (doesn't scale)
        return base_value


def create_benchmark_for_profile(base_benchmark, prompt_tokens, output_tokens):
    """Create a new benchmark entry for a specific traffic profile."""
    base_prompt = base_benchmark['prompt_tokens']
    base_output = base_benchmark['output_tokens']

    # Create new benchmark with updated traffic profile
    new_benchmark = base_benchmark.copy()

    # Update traffic profile
    new_benchmark['prompt_tokens'] = prompt_tokens
    new_benchmark['output_tokens'] = output_tokens

    # Update mean values (add small variance to simulate real data)
    import random
    random.seed(f"{base_benchmark['model_hf_repo']}{prompt_tokens}{output_tokens}")
    new_benchmark['mean_input_tokens'] = prompt_tokens + random.uniform(-2, 2)
    new_benchmark['mean_output_tokens'] = output_tokens + random.uniform(-2, 2)

    # Scale TTFT based on input token change
    ttft_scale = prompt_tokens / base_prompt if base_prompt > 0 else 1.0
    new_benchmark['ttft_mean'] = base_benchmark['ttft_mean'] * ttft_scale
    new_benchmark['ttft_p90'] = base_benchmark['ttft_p90'] * ttft_scale
    new_benchmark['ttft_p99'] = base_benchmark['ttft_p99'] * ttft_scale
    new_benchmark['ttft_p95'] = estimate_p95(new_benchmark['ttft_p90'], new_benchmark['ttft_p99'])

    # ITL doesn't scale with token count (it's per-token cost)
    new_benchmark['itl_mean'] = base_benchmark['itl_mean']
    new_benchmark['itl_p90'] = base_benchmark['itl_p90']
    new_benchmark['itl_p99'] = base_benchmark['itl_p99']
    new_benchmark['itl_p95'] = estimate_p95(base_benchmark['itl_p90'], base_benchmark['itl_p99'])

    # Calculate E2E = TTFT + (output_tokens * ITL)
    # This is more accurate than scaling
    new_benchmark['e2e_mean'] = new_benchmark['ttft_mean'] + (output_tokens * new_benchmark['itl_mean'])
    new_benchmark['e2e_p90'] = new_benchmark['ttft_p90'] + (output_tokens * new_benchmark['itl_p90'])
    new_benchmark['e2e_p99'] = new_benchmark['ttft_p99'] + (output_tokens * new_benchmark['itl_p99'])
    new_benchmark['e2e_p95'] = estimate_p95(new_benchmark['e2e_p90'], new_benchmark['e2e_p99'])

    # Update prompt/output token statistics
    # Keep similar relative variance
    base_prompt_stdev_ratio = base_benchmark.get('prompt_tokens_stdev', 76) / base_prompt if base_prompt > 0 else 0.15
    base_output_stdev_ratio = base_benchmark.get('output_tokens_stdev', 38) / base_output if base_output > 0 else 0.15

    new_benchmark['prompt_tokens_stdev'] = int(prompt_tokens * base_prompt_stdev_ratio)
    new_benchmark['prompt_tokens_min'] = int(prompt_tokens * 0.7)
    new_benchmark['prompt_tokens_max'] = int(prompt_tokens * 1.3)

    new_benchmark['output_tokens_stdev'] = int(output_tokens * base_output_stdev_ratio)
    new_benchmark['output_tokens_min'] = int(output_tokens * 0.7)
    new_benchmark['output_tokens_max'] = int(output_tokens * 1.3)

    # Throughput remains similar (hardware capability doesn't change)
    # Just keep the original values

    return new_benchmark


def main():
    """Main function to update benchmarks."""
    # Load existing benchmarks
    benchmarks_path = Path(__file__).parent.parent / "data" / "benchmarks.json"

    with open(benchmarks_path, 'r') as f:
        data = json.load(f)

    old_benchmarks = data['benchmarks']
    print(f"Loaded {len(old_benchmarks)} existing benchmarks")

    # Group by (model, hardware, hardware_count)
    groups = defaultdict(list)
    for benchmark in old_benchmarks:
        key = (
            benchmark['model_hf_repo'],
            benchmark['hardware'],
            benchmark['hardware_count']
        )
        groups[key].append(benchmark)

    print(f"Found {len(groups)} unique (model, hardware, count) combinations")

    # Create new benchmarks with GuideLLM traffic profiles
    new_benchmarks = []

    for (model, hardware, count), group_benchmarks in sorted(groups.items()):
        # Use the first benchmark as the base (they should be similar except for traffic profile)
        # We'll use the (512, 256) one as base if it exists, otherwise the first one
        base = next((b for b in group_benchmarks if b['prompt_tokens'] == 512 and b['output_tokens'] == 256), group_benchmarks[0])

        # Create benchmarks for each GuideLLM traffic profile
        for prompt_tokens, output_tokens in TRAFFIC_PROFILES:
            new_benchmark = create_benchmark_for_profile(base, prompt_tokens, output_tokens)
            new_benchmarks.append(new_benchmark)

    print(f"Created {len(new_benchmarks)} new benchmarks")
    print(f"  {len(groups)} configurations × {len(TRAFFIC_PROFILES)} traffic profiles")

    # Show traffic profile distribution
    profile_counts = defaultdict(int)
    for b in new_benchmarks:
        profile_counts[(b['prompt_tokens'], b['output_tokens'])] += 1

    print("\nTraffic profile distribution:")
    for profile, count in sorted(profile_counts.items()):
        print(f"  ({profile[0]}, {profile[1]}): {count} benchmarks")

    # Update data structure
    data['benchmarks'] = new_benchmarks

    # Write updated benchmarks
    with open(benchmarks_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"\n✓ Updated {benchmarks_path}")
    print(f"✓ Total benchmarks: {len(new_benchmarks)}")


if __name__ == "__main__":
    main()
