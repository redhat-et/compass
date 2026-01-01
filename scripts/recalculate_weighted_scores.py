#!/usr/bin/env python3
"""
Recalculate Weighted Scores for All Use Cases

Uses the interpolated benchmark data to calculate use-case specific
weighted scores for all 204 models.
"""

import pandas as pd
import numpy as np
import os

# Use-case specific weights for each benchmark
USE_CASE_WEIGHTS = {
    "chatbot_conversational": {
        "mmlu_pro": 0.15,
        "tau2": 0.20,
        "ifbench": 0.15,
        "aime_25": 0.15,
        "artificial_analysis_intelligence_index": 0.15,
        "gpqa": 0.10,
        "hle": 0.10,
    },
    "code_completion": {
        "artificial_analysis_coding_index": 0.25,
        "livecodebench": 0.25,
        "terminalbench_hard": 0.15,
        "aime_25": 0.15,
        "artificial_analysis_math_index": 0.10,
        "tau2": 0.10,
    },
    "code_generation_detailed": {
        "artificial_analysis_coding_index": 0.20,
        "livecodebench": 0.20,
        "aime_25": 0.20,
        "artificial_analysis_math_index": 0.15,
        "terminalbench_hard": 0.15,
        "tau2": 0.10,
    },
    "translation": {
        "mmlu_pro": 0.25,
        "artificial_analysis_intelligence_index": 0.20,
        "ifbench": 0.20,
        "tau2": 0.15,
        "aime_25": 0.10,
        "gpqa": 0.10,
    },
    "content_generation": {
        "mmlu_pro": 0.20,
        "artificial_analysis_intelligence_index": 0.20,
        "ifbench": 0.15,
        "tau2": 0.15,
        "aime_25": 0.15,
        "gpqa": 0.10,
        "hle": 0.05,
    },
    "summarization_short": {
        "mmlu_pro": 0.20,
        "ifbench": 0.20,
        "tau2": 0.20,
        "artificial_analysis_intelligence_index": 0.15,
        "aime_25": 0.15,
        "gpqa": 0.10,
    },
    "document_analysis_rag": {
        "mmlu_pro": 0.20,
        "tau2": 0.25,
        "gpqa": 0.15,
        "aime_25": 0.15,
        "artificial_analysis_intelligence_index": 0.15,
        "ifbench": 0.10,
    },
    "long_document_summarization": {
        "mmlu_pro": 0.15,
        "tau2": 0.25,
        "ifbench": 0.15,
        "aime_25": 0.15,
        "gpqa": 0.15,
        "artificial_analysis_intelligence_index": 0.15,
    },
    "research_legal_analysis": {
        "gpqa": 0.20,
        "aime_25": 0.20,
        "artificial_analysis_math_index": 0.15,
        "mmlu_pro": 0.15,
        "tau2": 0.15,
        "artificial_analysis_intelligence_index": 0.15,
    },
}

def clean_percentage(val):
    """Convert percentage string to float."""
    if pd.isna(val) or val == 'N/A' or val == '':
        return 0.0
    if isinstance(val, str):
        return float(val.replace('%', ''))
    return float(val)

def calculate_weighted_score(row, weights):
    """Calculate weighted score for a model."""
    total = 0.0
    total_weight = 0.0
    
    for benchmark, weight in weights.items():
        if benchmark in row.index:
            score = clean_percentage(row[benchmark])
            if score > 0:
                total += score * weight
                total_weight += weight
    
    # Normalize if not all benchmarks available
    if total_weight > 0:
        return total / total_weight * (total_weight / sum(weights.values()))
    return 0.0

def main():
    print("=" * 70)
    print("RECALCULATING WEIGHTED SCORES FOR ALL USE CASES")
    print("=" * 70)
    
    # Load interpolated data
    df = pd.read_csv('data/benchmarks/models/opensource_all_benchmarks_interpolated.csv')
    print(f"Loaded {len(df)} models with complete benchmark data")
    
    # Output directory
    output_dir = 'data/business_context/use_case/weighted_scores'
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate scores for each use case
    for use_case, weights in USE_CASE_WEIGHTS.items():
        print(f"\nProcessing: {use_case}")
        
        scores = []
        for _, row in df.iterrows():
            score = calculate_weighted_score(row, weights)
            scores.append({
                'model_name': row['Model Name'],
                'provider': row['Provider'],
                'dataset': row.get('Dataset', 'Unknown'),
                'weighted_score': f"{score:.2f}%"
            })
        
        # Create output dataframe
        output_df = pd.DataFrame(scores)
        output_df = output_df.sort_values('weighted_score', ascending=False)
        
        # Save to CSV
        output_path = os.path.join(output_dir, f'opensource_{use_case}.csv')
        output_df.to_csv(output_path, index=False)
        
        # Show top 5
        print(f"  Top 5 for {use_case}:")
        for i, row in output_df.head(5).iterrows():
            print(f"    {row['model_name'][:40]:<40}: {row['weighted_score']}")
    
    print("\n" + "=" * 70)
    print("✅ All use-case weighted scores recalculated!")
    print(f"Output: {output_dir}/opensource_*.csv")
    print("=" * 70)
    
    # Show comparison for Qwen3 235B
    print("\n" + "=" * 70)
    print("QWEN3 235B A22B 2507 (REASONING) - NEW SCORES")
    print("=" * 70)
    
    qwen_row = df[df['Model Name'].str.contains('Qwen3 235B A22B 2507', case=False, na=False)]
    if len(qwen_row) > 0:
        for use_case, weights in USE_CASE_WEIGHTS.items():
            score = calculate_weighted_score(qwen_row.iloc[0], weights)
            print(f"  {use_case:<30}: {score:.1f}%")

if __name__ == '__main__':
    main()

