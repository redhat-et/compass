#!/usr/bin/env python3
"""
Regenerate synthetic benchmark data from exported PSV file.

This script:
1. Reads real benchmarks from data/real_benchmarks.psv
2. Applies random variation (±15%) to TTFT and ITL values
3. Recalculates E2E from TTFT + (ITL × output_tokens)
4. Writes to data/benchmarks.json
"""

import json
import random
from pathlib import Path


def apply_random_variation(value, variation_pct=15):
    """Apply random variation within ±variation_pct%."""
    if value is None or value == 0 or value == '':
        return value

    value = float(value)
    variation = random.uniform(-variation_pct, variation_pct) / 100.0
    return value * (1 + variation)


def parse_psv_line(line):
    """Parse a PSV line into a tuple."""
    parts = line.strip().split('|')
    if len(parts) != 25:
        return None

    return tuple(parts)


def generate_synthetic_benchmark(real_bench_tuple):
    """Generate synthetic benchmark with random variation from real data."""
    (
        model_hf_repo, hardware, hardware_count, framework, framework_version,
        mean_input_tokens, mean_output_tokens,
        ttft_mean, ttft_p90, ttft_p95, ttft_p99,
        itl_mean, itl_p90, itl_p95, itl_p99,
        e2e_mean, e2e_p90, e2e_p95, e2e_p99,
        requests_per_second, tokens_per_second,
        prompt_tokens, prompt_tokens_stdev,
        output_tokens, output_tokens_stdev
    ) = real_bench_tuple

    # Convert to float
    mean_input_tokens = float(mean_input_tokens)
    mean_output_tokens = float(mean_output_tokens)
    hardware_count = int(hardware_count)

    # Apply ±15% random variation to TTFT and ITL
    ttft_mean_synth = apply_random_variation(ttft_mean, 15)
    ttft_p90_synth = apply_random_variation(ttft_p90, 15)
    ttft_p95_synth = apply_random_variation(ttft_p95, 15)
    ttft_p99_synth = apply_random_variation(ttft_p99, 15)

    itl_mean_synth = apply_random_variation(itl_mean, 15)
    itl_p90_synth = apply_random_variation(itl_p90, 15)
    itl_p95_synth = apply_random_variation(itl_p95, 15)
    itl_p99_synth = apply_random_variation(itl_p99, 15)

    # Recalculate E2E from TTFT + (ITL × output_tokens)
    e2e_mean_synth = ttft_mean_synth + (itl_mean_synth * mean_output_tokens)
    e2e_p90_synth = ttft_p90_synth + (itl_p90_synth * mean_output_tokens)
    e2e_p95_synth = ttft_p95_synth + (itl_p95_synth * mean_output_tokens)
    e2e_p99_synth = ttft_p99_synth + (itl_p99_synth * mean_output_tokens)

    # Apply variation to QPS and TPS
    requests_per_second_synth = apply_random_variation(requests_per_second, 15)
    tokens_per_second_synth = apply_random_variation(tokens_per_second, 15) if tokens_per_second and float(tokens_per_second) > 0 else None

    # Build synthetic benchmark record
    return {
        "model_hf_repo": model_hf_repo,
        "hardware": hardware,
        "hardware_count": hardware_count,
        "framework": framework,
        "framework_version": framework_version if framework_version else None,
        "mean_input_tokens": float(mean_input_tokens),
        "mean_output_tokens": float(mean_output_tokens),
        "prompt_tokens": float(prompt_tokens) if prompt_tokens else float(mean_input_tokens),
        "prompt_tokens_stdev": float(prompt_tokens_stdev) if prompt_tokens_stdev else 0.0,
        "output_tokens": float(output_tokens) if output_tokens else float(mean_output_tokens),
        "output_tokens_stdev": float(output_tokens_stdev) if output_tokens_stdev else 0.0,
        "ttft_mean": round(ttft_mean_synth, 2),
        "ttft_p90": round(ttft_p90_synth, 2),
        "ttft_p95": round(ttft_p95_synth, 2),
        "ttft_p99": round(ttft_p99_synth, 2),
        "itl_mean": round(itl_mean_synth, 2),
        "itl_p90": round(itl_p90_synth, 2),
        "itl_p95": round(itl_p95_synth, 2),
        "itl_p99": round(itl_p99_synth, 2),
        "e2e_mean": round(e2e_mean_synth, 2),
        "e2e_p90": round(e2e_p90_synth, 2),
        "e2e_p95": round(e2e_p95_synth, 2),
        "e2e_p99": round(e2e_p99_synth, 2),
        "requests_per_second": round(requests_per_second_synth, 2),
        "tokens_per_second": round(tokens_per_second_synth, 2) if tokens_per_second_synth else None,
    }


def main():
    """Main function."""
    print("=" * 70)
    print("Regenerating Realistic Synthetic Benchmarks from PSV Export")
    print("=" * 70)
    print()

    # Read PSV file
    psv_path = Path(__file__).parent.parent / "data" / "real_benchmarks.psv"
    print(f"Reading from {psv_path}...")

    real_benchmarks = []
    with open(psv_path, 'r') as f:
        for line in f:
            parsed = parse_psv_line(line)
            if parsed:
                real_benchmarks.append(parsed)

    print(f"✓ Read {len(real_benchmarks)} real benchmark configurations")
    print()

    # Generate synthetic benchmarks with variation
    print("Generating synthetic benchmarks with ±15% variation...")
    synthetic_benchmarks = []

    for real_bench in real_benchmarks:
        synth_bench = generate_synthetic_benchmark(real_bench)
        synthetic_benchmarks.append(synth_bench)

    print(f"✓ Generated {len(synthetic_benchmarks)} synthetic benchmarks")
    print()

    # Statistics
    models = set(b["model_hf_repo"] for b in synthetic_benchmarks)
    hardware_types = set(b["hardware"] for b in synthetic_benchmarks)
    traffic_profiles = set((b["mean_input_tokens"], b["mean_output_tokens"]) for b in synthetic_benchmarks)

    print("📊 Synthetic Data Statistics:")
    print(f"  Models: {len(models)}")
    print(f"  Hardware types: {len(hardware_types)}")
    print(f"  Traffic profiles: {len(traffic_profiles)}")
    print(f"  Total benchmarks: {len(synthetic_benchmarks)}")
    print()

    print("🖥️  Hardware types:")
    for hw in sorted(hardware_types):
        count = sum(1 for b in synthetic_benchmarks if b["hardware"] == hw)
        print(f"  {hw}: {count} benchmarks")
    print()

    print("🚦 Traffic profiles:")
    for tp in sorted(traffic_profiles):
        count = sum(1 for b in synthetic_benchmarks if (b["mean_input_tokens"], b["mean_output_tokens"]) == tp)
        print(f"  {int(tp[0])}→{int(tp[1])}: {count} benchmarks")
    print()

    # Show sample benchmarks for Llama 3.1 8B
    print("📋 Sample: Llama 3.1 8B on H100 (1024→1024):")
    samples = [b for b in synthetic_benchmarks
               if 'Llama-3.1-8B' in b['model_hf_repo']
               and b['hardware'] == 'H100'
               and b['mean_input_tokens'] == 1024
               and b['mean_output_tokens'] == 1024]

    for s in sorted(samples, key=lambda x: x['hardware_count']):
        print(f"  {s['hardware_count']}x {s['hardware']}: TTFT p95={s['ttft_p95']:.1f}ms, ITL p95={s['itl_p95']:.1f}ms, E2E p95={s['e2e_p95']:.1f}ms, QPS={s['requests_per_second']:.2f}")
    print()

    # Write to file
    output_path = Path(__file__).parent.parent / "data" / "benchmarks.json"
    print(f"Writing to {output_path}...")

    data = {
        "_metadata": {
            "description": "Synthetic benchmark data for development and testing",
            "version": "3.0-realistic",
            "schema_changes": [
                "Generated from real PostgreSQL benchmarks with ±15% random variation",
                "Hardware names match real data (H100, H200, NVIDIA-A100-40GB, etc.)",
                "Performance values realistic and vary by model/GPU/tensor_parallel",
                "E2E calculated as: TTFT + (ITL × output_tokens)",
                "QPS varies realistically (not all 10.0)"
            ],
            "generation_method": "scripts/regenerate_from_psv.py",
            "source": "PostgreSQL exported_summaries table (via data/real_benchmarks.psv)",
            "variation": "±15% random variation applied to TTFT and ITL",
            "random_seed": 42
        },
        "benchmarks": synthetic_benchmarks
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"✓ Wrote {len(synthetic_benchmarks)} benchmarks")
    print()

    print("=" * 70)
    print("✅ Regeneration complete!")
    print("=" * 70)
    print()
    print("Next steps:")
    print("  scripts/load_benchmarks.py  # Load into PostgreSQL")
    print("  make backend-start           # Start backend")
    print("  make ui-start                # Start UI and test")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    main()
