#!/bin/bash

# Start the NeuralNav FastAPI backend
# This script installs dependencies with uv and starts the API server

set -e

echo "🚀 Starting NeuralNav API..."
echo ""

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "❌ uv not found! Install it: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

# Install/sync dependencies
echo "Syncing dependencies..."
uv sync
echo "✅ Dependencies ready"
echo ""

# Check if Ollama is running
echo "Checking if Ollama is running..."
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "⚠️  Warning: Ollama is not running on http://localhost:11434"
    echo "Please start it: ollama serve"
    echo ""
fi

# Start FastAPI
echo "Starting FastAPI backend on http://localhost:8000..."
echo ""
cd backend
uv run uvicorn src.api.routes:app --host 0.0.0.0 --port 8000 --reload
