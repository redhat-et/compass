#!/bin/bash

# Start the AI Pre-Deployment Assistant UI
# This script activates the virtual environment and starts the Streamlit UI

set -e

echo "🤖 Starting AI Pre-Deployment Assistant UI..."
echo ""

# Check if virtual environment exists
if [ ! -d "backend/venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run: cd backend && python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source backend/venv/bin/activate

# Check if FastAPI backend is running
echo "Checking if FastAPI backend is running..."
if ! curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "⚠️  Warning: FastAPI backend is not running on http://localhost:8000"
    echo "Please start it in another terminal: ./run_api.sh"
    echo ""
fi

# Start Streamlit
echo "Starting Streamlit UI on http://localhost:8501..."
echo ""
streamlit run ui/app.py
