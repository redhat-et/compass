FROM python:3.11-slim

WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy simulator service
COPY simulator_service.py .

# Create data directory for benchmarks
RUN mkdir -p /app/data

# Copy benchmark data (will be mounted or copied at build time)
COPY data/benchmarks.json /app/data/benchmarks.json

# Expose vLLM-compatible port
EXPOSE 8080

# Set default environment variables
ENV MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
ENV TENSOR_PARALLEL_SIZE="1"
ENV GPU_TYPE="NVIDIA-L4"
ENV PORT="8080"

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8080/health')" || exit 1

# Run the simulator
CMD ["python", "simulator_service.py"]
