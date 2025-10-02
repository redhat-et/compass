#!/bin/bash

# Start the AI Pre-Deployment Assistant FastAPI backend
# This script activates the virtual environment and starts the API server

set -e

echo "🚀 Starting AI Pre-Deployment Assistant API..."
echo ""

# Check if virtual environment exists
if [ ! -d "backend/venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: cd backend && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source backend/venv/bin/activate

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
uvicorn src.api.routes:app --host 0.0.0.0 --port 8000 --reload
