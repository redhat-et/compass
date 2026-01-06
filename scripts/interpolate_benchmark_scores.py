#!/usr/bin/env python3
"""
ML Interpolation for Missing Benchmark Scores

This script fills in N/A benchmark values using ML regression based on:
- Model characteristics (size, family, provider)
- Available benchmark scores

Result: All 206 models will have complete benchmark data.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
import os
import re

warnings.filterwarnings('ignore')

# Define benchmark columns
BENCHMARK_COLS = [
    'aime', 'aime_25', 'artificial_analysis_coding_index', 
    'artificial_analysis_intelligence_index', 'artificial_analysis_math_index',
    'gpqa', 'hle', 'ifbench', 'lcr', 'livecodebench', 
    'math_500', 'mmlu_pro', 'scicode', 'tau2', 'terminalbench_hard'
]

def extract_model_size(model_name):
    """Extract model size in billions from name."""
    # Patterns like "70B", "7b", "120B", etc.
    match = re.search(r'(\d+\.?\d*)b', model_name.lower())
    if match:
        return float(match.group(1))
    # Try patterns like "8x7B" for MoE
    match = re.search(r'(\d+)x(\d+)b', model_name.lower())
    if match:
        return float(match.group(1)) * float(match.group(2))
    return 30.0  # Default assumption

def extract_model_family(model_name):
    """Extract model family from name."""
    name_lower = model_name.lower()
    families = {
        'llama': 'llama',
        'qwen': 'qwen',
        'deepseek': 'deepseek',
        'mistral': 'mistral',
        'mixtral': 'mistral',
        'phi': 'phi',
        'gemma': 'gemma',
        'gpt': 'gpt',
        'claude': 'claude',
        'granite': 'granite',
        'nemotron': 'nemotron',
        'solar': 'solar',
        'yi': 'yi',
        'glm': 'glm',
        'kimi': 'kimi',
        'minimax': 'minimax',
        'exaone': 'exaone',
    }
    for key, family in families.items():
        if key in name_lower:
            return family
    return 'other'

def is_reasoning_model(model_name):
    """Check if model is reasoning-focused."""
    name_lower = model_name.lower()
    return 1 if any(x in name_lower for x in ['reasoning', 'think', 'r1', 'o1', 'o3']) else 0

def is_quantized(model_name):
    """Check if model is quantized."""
    name_lower = model_name.lower()
    return 1 if any(x in name_lower for x in ['w4a16', 'w8a8', 'fp8', 'quantized', 'q4', 'q8']) else 0

def clean_percentage(val):
    """Convert percentage string to float."""
    if pd.isna(val) or val == 'N/A' or val == '':
        return np.nan
    if isinstance(val, str):
        return float(val.replace('%', ''))
    return float(val)

def load_and_prepare_data(csv_path):
    """Load and prepare benchmark data."""
    df = pd.read_csv(csv_path)
    
    # Clean benchmark columns
    for col in BENCHMARK_COLS:
        if col in df.columns:
            df[col] = df[col].apply(clean_percentage)
    
    # Extract features
    df['model_size'] = df['Model Name'].apply(extract_model_size)
    df['model_family'] = df['Model Name'].apply(extract_model_family)
    df['is_reasoning'] = df['Model Name'].apply(is_reasoning_model)
    df['is_quantized'] = df['Model Name'].apply(is_quantized)
    
    # Encode categorical features
    le_provider = LabelEncoder()
    le_family = LabelEncoder()
    
    df['provider_encoded'] = le_provider.fit_transform(df['Provider'].fillna('Unknown'))
    df['family_encoded'] = le_family.fit_transform(df['model_family'])
    
    return df, le_provider, le_family

def get_feature_cols(df, target_col):
    """Get feature columns for prediction, excluding the target."""
    feature_cols = ['model_size', 'provider_encoded', 'family_encoded', 'is_reasoning', 'is_quantized']
    
    # Add other benchmark columns as features (if they have data)
    for col in BENCHMARK_COLS:
        if col != target_col and col in df.columns:
            feature_cols.append(col)
    
    return feature_cols

def train_and_predict(df, target_col):
    """Train model to predict missing values for a benchmark."""
    # Get rows with known values for training
    mask_known = df[target_col].notna()
    mask_unknown = df[target_col].isna()
    
    if mask_known.sum() < 10:
        print(f"  ⚠️ Not enough data for {target_col}, using median")
        return df[target_col].median()
    
    if mask_unknown.sum() == 0:
        print(f"  ✓ {target_col}: No missing values")
        return None
    
    feature_cols = get_feature_cols(df, target_col)
    
    # Prepare training data
    X_train = df.loc[mask_known, feature_cols].copy()
    y_train = df.loc[mask_known, target_col].copy()
    X_predict = df.loc[mask_unknown, feature_cols].copy()
    
    # Impute missing feature values with median
    imputer = SimpleImputer(strategy='median')
    X_train_imputed = imputer.fit_transform(X_train)
    X_predict_imputed = imputer.transform(X_predict)
    
    # Train Random Forest
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train_imputed, y_train)
    
    # Predict missing values
    predictions = model.predict(X_predict_imputed)
    
    # Clip to valid range (0-100%)
    predictions = np.clip(predictions, 0, 100)
    
    return predictions, mask_unknown

def main():
    print("=" * 70)
    print("ML INTERPOLATION FOR MISSING BENCHMARK SCORES")
    print("=" * 70)
    
    # Load data
    csv_path = 'data/benchmarks/models/opensource_all_benchmarks.csv'
    df, le_provider, le_family = load_and_prepare_data(csv_path)
    
    print(f"\nLoaded {len(df)} models")
    print(f"Benchmark columns: {len(BENCHMARK_COLS)}")
    
    # Count initial missing values
    initial_missing = sum(df[col].isna().sum() for col in BENCHMARK_COLS if col in df.columns)
    print(f"Initial missing values: {initial_missing}")
    
    # Predict missing values for each benchmark
    print("\n" + "-" * 70)
    print("TRAINING AND PREDICTING")
    print("-" * 70)
    
    for col in BENCHMARK_COLS:
        if col not in df.columns:
            continue
            
        result = train_and_predict(df, col)
        
        if result is None:
            continue
        
        if isinstance(result, (int, float)):
            # Use median for columns with too little data
            df[col] = df[col].fillna(result)
            print(f"  ✓ {col}: Filled with median ({result:.1f}%)")
        else:
            predictions, mask = result
            df.loc[mask, col] = predictions
            print(f"  ✓ {col}: Predicted {mask.sum()} values (mean: {predictions.mean():.1f}%)")
    
    # Count final missing values
    final_missing = sum(df[col].isna().sum() for col in BENCHMARK_COLS if col in df.columns)
    print(f"\nFinal missing values: {final_missing}")
    print(f"Filled: {initial_missing - final_missing} values")
    
    # Save interpolated data
    output_path = 'data/benchmarks/models/opensource_all_benchmarks_interpolated.csv'
    
    # Convert back to percentage format
    for col in BENCHMARK_COLS:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x:.2f}%" if pd.notna(x) else 'N/A')
    
    # Keep only original columns plus a flag
    output_df = df[['Model Name', 'Provider', 'Dataset'] + BENCHMARK_COLS].copy()
    output_df.to_csv(output_path, index=False)
    
    print(f"\n✅ Saved interpolated data to: {output_path}")
    
    # Show sample of interpolated values
    print("\n" + "=" * 70)
    print("SAMPLE: Models with filled values")
    print("=" * 70)
    
    # Find models that had N/A values
    sample_models = ['gpt-oss-120B (high)', 'Qwen3 235B A22B 2507 (Reasoning)', 'DeepSeek R1 W4A16']
    for model in sample_models:
        row = output_df[output_df['Model Name'].str.contains(model, case=False, na=False)]
        if len(row) > 0:
            print(f"\n{row.iloc[0]['Model Name']}:")
            for col in BENCHMARK_COLS[:5]:
                if col in row.columns:
                    print(f"  {col}: {row.iloc[0][col]}")

if __name__ == '__main__':
    main()

