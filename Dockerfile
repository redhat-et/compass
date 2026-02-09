# Backend Dockerfile for NeuralNav
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python dependencies (frozen = use lockfile exactly, no-dev = skip dev deps)
RUN uv sync --frozen --no-dev

# Copy backend source code
COPY src/neuralnav ./src/neuralnav

# Copy data files (Knowledge Base)
COPY data ./data

# Create directories for generated files
RUN mkdir -p /app/generated_configs /app/logs

# Set environment variables
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Expose backend API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD uv run python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the backend API server
CMD ["uv", "run", "uvicorn", "neuralnav.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
