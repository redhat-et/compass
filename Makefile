# AI Pre-Deployment Assistant - Makefile
#
# This Makefile provides common development tasks for the AI Pre-Deployment Assistant.
# Supports macOS and Linux.

.PHONY: help
.DEFAULT_GOAL := help

# Platform detection
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

ifeq ($(UNAME_S),Darwin)
    PLATFORM := macos
    OPEN_CMD := open
    PYTHON := python3
else ifeq ($(UNAME_S),Linux)
    PLATFORM := linux
    OPEN_CMD := xdg-open
    PYTHON := python3
else
    $(error Unsupported platform: $(UNAME_S). Please use macOS or Linux (or WSL2 on Windows))
endif

# Configuration
REGISTRY ?= quay.io
REGISTRY_ORG ?= vllm-assistant
SIMULATOR_IMAGE ?= vllm-simulator
SIMULATOR_TAG ?= latest
SIMULATOR_FULL_IMAGE := $(REGISTRY)/$(REGISTRY_ORG)/$(SIMULATOR_IMAGE):$(SIMULATOR_TAG)

OLLAMA_MODEL ?= llama3.1:8b
KIND_CLUSTER_NAME ?= ai-assistant-poc

BACKEND_DIR := backend
UI_DIR := ui
SIMULATOR_DIR := simulator

BACKEND_VENV := $(BACKEND_DIR)/venv
# UI uses backend's venv (no separate venv)

# PID files for background processes
PID_DIR := .pids
OLLAMA_PID := $(PID_DIR)/ollama.pid
BACKEND_PID := $(PID_DIR)/backend.pid
UI_PID := $(PID_DIR)/ui.pid

# Log directory
LOG_DIR := logs

# Allow local overrides via .env file
-include .env
export

# Colors for output
RED := \033[0;31m
GREEN := \033[0;32m
YELLOW := \033[1;33m
BLUE := \033[0;34m
NC := \033[0m # No Color

##@ Help

help: ## Display this help message
	@awk 'BEGIN {FS = ":.*##"; printf "\n$(BLUE)Usage:$(NC)\n  make $(GREEN)<target>$(NC)\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2 } /^##@/ { printf "\n$(BLUE)%s$(NC)\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup & Installation

check-prereqs: ## Check if required tools are installed
	@echo "$(BLUE)Checking prerequisites...$(NC)"
	@command -v docker >/dev/null 2>&1 || (echo "$(RED)✗ docker not found$(NC). Install from https://www.docker.com/products/docker-desktop" && exit 1)
	@echo "$(GREEN)✓ docker found$(NC)"
	@command -v kubectl >/dev/null 2>&1 || (echo "$(RED)✗ kubectl not found$(NC). Run: brew install kubectl" && exit 1)
	@echo "$(GREEN)✓ kubectl found$(NC)"
	@command -v kind >/dev/null 2>&1 || (echo "$(RED)✗ kind not found$(NC). Run: brew install kind" && exit 1)
	@echo "$(GREEN)✓ kind found$(NC)"
	@command -v ollama >/dev/null 2>&1 || (echo "$(RED)✗ ollama not found$(NC). Run: brew install ollama" && exit 1)
	@echo "$(GREEN)✓ ollama found$(NC)"
	@command -v $(PYTHON) >/dev/null 2>&1 || (echo "$(RED)✗ $(PYTHON) not found$(NC). Install Python 3.11+" && exit 1)
	@echo "$(GREEN)✓ $(PYTHON) found$(NC)"
	@docker info >/dev/null 2>&1 || (echo "$(RED)✗ Docker daemon not running$(NC). Start Docker Desktop" && exit 1)
	@echo "$(GREEN)✓ Docker daemon running$(NC)"
	@echo "$(GREEN)All prerequisites satisfied!$(NC)"

setup-backend: ## Set up backend Python environment (includes UI dependencies)
	@echo "$(BLUE)Setting up backend environment...$(NC)"
	cd $(BACKEND_DIR) && $(PYTHON) -m venv venv
	cd $(BACKEND_DIR) && . venv/bin/activate && pip install --upgrade pip
	cd $(BACKEND_DIR) && . venv/bin/activate && pip install -r requirements.txt
	@echo "$(GREEN)✓ Backend environment ready (includes UI dependencies)$(NC)"

setup-ui: setup-backend ## Set up UI (uses backend's venv)
	@echo "$(GREEN)✓ UI ready (shares backend venv)$(NC)"

setup-ollama: ## Pull Ollama model
	@echo "$(BLUE)Checking if Ollama model $(OLLAMA_MODEL) is available...$(NC)"
	@# Start ollama if not running
	@if ! pgrep -x "ollama" > /dev/null; then \
		echo "$(YELLOW)Starting Ollama service...$(NC)"; \
		ollama serve > /dev/null 2>&1 & \
		sleep 2; \
	fi
	@# Check if model exists, pull if not
	@ollama list | grep -q $(OLLAMA_MODEL) || (echo "$(YELLOW)Pulling model $(OLLAMA_MODEL)...$(NC)" && ollama pull $(OLLAMA_MODEL))
	@echo "$(GREEN)✓ Ollama model $(OLLAMA_MODEL) ready$(NC)"

setup: check-prereqs setup-backend setup-ui setup-ollama ## Run all setup tasks
	@echo "$(GREEN)✓ Setup complete!$(NC)"
	@echo ""
	@echo "$(BLUE)Next steps:$(NC)"
	@echo "  make cluster-start # Create Kubernetes cluster"
	@echo "  make dev           # Start all services"

##@ Development

dev: setup-ollama ## Start all services (Ollama + Backend + UI)
	@echo "$(BLUE)Starting all services...$(NC)"
	@mkdir -p $(PID_DIR)
	@$(MAKE) start-ollama
	@sleep 3
	@$(MAKE) start-backend
	@sleep 3
	@$(MAKE) start-ui
	@echo ""
	@echo "$(GREEN)✓ All services started!$(NC)"
	@echo ""
	@echo "$(BLUE)Service URLs:$(NC)"
	@echo "  UI:      http://localhost:8501"
	@echo "  Backend: http://localhost:8000"
	@echo "  Ollama:  http://localhost:11434"
	@echo ""
	@echo "$(BLUE)Logs:$(NC)"
	@echo "  make logs-backend"
	@echo "  make logs-ui"
	@echo ""
	@echo "$(BLUE)Stop:$(NC)"
	@echo "  make stop"

start-ollama: ## Start Ollama service
	@echo "$(BLUE)Starting Ollama...$(NC)"
	@if pgrep -x "ollama" > /dev/null; then \
		echo "$(YELLOW)Ollama already running$(NC)"; \
	else \
		ollama serve > /dev/null 2>&1 & echo $$! > $(OLLAMA_PID); \
		echo "$(GREEN)✓ Ollama started (PID: $$(cat $(OLLAMA_PID)))$(NC)"; \
	fi

start-backend: ## Start FastAPI backend
	@echo "$(BLUE)Starting backend...$(NC)"
	@mkdir -p $(PID_DIR) $(LOG_DIR)
	@if [ -f $(BACKEND_PID) ] && kill -0 $$(cat $(BACKEND_PID)) 2>/dev/null; then \
		echo "$(YELLOW)Backend already running (PID: $$(cat $(BACKEND_PID)))$(NC)"; \
	else \
		cd $(BACKEND_DIR) && . venv/bin/activate && \
		( uvicorn src.api.routes:app --reload --host 0.0.0.0 --port 8000 > ../$(LOG_DIR)/backend.log 2>&1 & echo $$! > ../$(BACKEND_PID) ); \
		sleep 2; \
		echo "$(GREEN)✓ Backend started (PID: $$(cat $(BACKEND_PID)))$(NC)"; \
	fi

start-ui: ## Start Streamlit UI
	@echo "$(BLUE)Starting UI...$(NC)"
	@mkdir -p $(PID_DIR) $(LOG_DIR)
	@if [ -f $(UI_PID) ] && kill -0 $$(cat $(UI_PID)) 2>/dev/null; then \
		echo "$(YELLOW)UI already running (PID: $$(cat $(UI_PID)))$(NC)"; \
	else \
		. $(BACKEND_DIR)/venv/bin/activate && streamlit run $(UI_DIR)/app.py --server.headless true > $(LOG_DIR)/ui.log 2>&1 & echo $$! > $(UI_PID); \
		sleep 2; \
		echo "$(GREEN)✓ UI started (PID: $$(cat $(UI_PID)))$(NC)"; \
	fi

stop: ## Stop all services
	@echo "$(BLUE)Stopping services...$(NC)"
	@if [ -f $(UI_PID) ]; then \
		kill $$(cat $(UI_PID)) 2>/dev/null || true; \
		rm -f $(UI_PID); \
		echo "$(GREEN)✓ UI stopped$(NC)"; \
	fi
	@if [ -f $(BACKEND_PID) ]; then \
		kill $$(cat $(BACKEND_PID)) 2>/dev/null || true; \
		rm -f $(BACKEND_PID); \
		echo "$(GREEN)✓ Backend stopped$(NC)"; \
	fi
	@# Don't stop Ollama as it might be used by other apps
	@echo "$(YELLOW)Note: Ollama left running (use 'pkill ollama' to stop manually)$(NC)"

restart: stop dev ## Restart all services

logs-backend: ## Tail backend logs
	@tail -f $(BACKEND_PID).log

logs-ui: ## Tail UI logs
	@tail -f $(UI_PID).log

health: ## Check if all services are running
	@echo "$(BLUE)Checking service health...$(NC)"
	@curl -s http://localhost:8000/health > /dev/null && echo "$(GREEN)✓ Backend healthy$(NC)" || echo "$(RED)✗ Backend not responding$(NC)"
	@curl -s http://localhost:8501 > /dev/null && echo "$(GREEN)✓ UI healthy$(NC)" || echo "$(RED)✗ UI not responding$(NC)"
	@curl -s http://localhost:11434/api/tags > /dev/null && echo "$(GREEN)✓ Ollama healthy$(NC)" || echo "$(RED)✗ Ollama not responding$(NC)"

open-ui: ## Open UI in browser
	@$(OPEN_CMD) http://localhost:8501

open-backend: ## Open backend API docs in browser
	@$(OPEN_CMD) http://localhost:8000/docs

##@ Docker & Simulator

build-simulator: ## Build vLLM simulator Docker image
	@echo "$(BLUE)Building simulator image...$(NC)"
	cd $(SIMULATOR_DIR) && docker build -t vllm-simulator:latest -t $(SIMULATOR_FULL_IMAGE) .
	@echo "$(GREEN)✓ Simulator image built:$(NC)"
	@echo "  - vllm-simulator:latest"
	@echo "  - $(SIMULATOR_FULL_IMAGE)"

push-simulator: build-simulator ## Push simulator image to Quay.io
	@echo "$(BLUE)Pushing simulator image to $(SIMULATOR_FULL_IMAGE)...$(NC)"
	@# Check if logged in to Quay.io
	@if ! docker login quay.io --get-login > /dev/null 2>&1; then \
		echo "$(YELLOW)Not logged in to Quay.io. Please login:$(NC)"; \
		docker login quay.io || (echo "$(RED)✗ Login failed$(NC)" && exit 1); \
	else \
		echo "$(GREEN)✓ Already logged in to Quay.io$(NC)"; \
	fi
	@echo "$(BLUE)Pushing image...$(NC)"
	docker push $(SIMULATOR_FULL_IMAGE)
	@echo "$(GREEN)✓ Image pushed to $(SIMULATOR_FULL_IMAGE)$(NC)"

pull-simulator: ## Pull simulator image from Quay.io
	@echo "$(BLUE)Pulling simulator image from $(SIMULATOR_FULL_IMAGE)...$(NC)"
	docker pull $(SIMULATOR_FULL_IMAGE)
	docker tag $(SIMULATOR_FULL_IMAGE) vllm-simulator:latest
	@echo "$(GREEN)✓ Image pulled and tagged as vllm-simulator:latest$(NC)"

##@ Kubernetes Cluster

cluster-start: check-prereqs build-simulator ## Create KIND cluster and load simulator image
	@echo "$(BLUE)Creating KIND cluster...$(NC)"
	./scripts/kind-cluster.sh start
	@echo "$(GREEN)✓ Cluster ready!$(NC)"

cluster-stop: ## Delete KIND cluster
	@echo "$(BLUE)Stopping KIND cluster...$(NC)"
	./scripts/kind-cluster.sh stop
	@echo "$(GREEN)✓ Cluster deleted$(NC)"

cluster-restart: ## Restart KIND cluster
	@echo "$(BLUE)Restarting KIND cluster...$(NC)"
	./scripts/kind-cluster.sh restart
	@echo "$(GREEN)✓ Cluster restarted$(NC)"

cluster-status: ## Show cluster status
	./scripts/kind-cluster.sh status

cluster-load-simulator: build-simulator ## Load simulator image into KIND cluster
	@echo "$(BLUE)Loading simulator image into KIND cluster...$(NC)"
	kind load docker-image vllm-simulator:latest --name $(KIND_CLUSTER_NAME)
	@echo "$(GREEN)✓ Simulator image loaded$(NC)"

clean-deployments: ## Delete all InferenceServices from cluster
	@echo "$(BLUE)Deleting all InferenceServices...$(NC)"
	kubectl delete inferenceservices --all
	@echo "$(GREEN)✓ All deployments deleted$(NC)"

##@ Testing

test: test-unit ## Run all tests
	@echo "$(GREEN)✓ All tests passed$(NC)"

test-unit: ## Run unit tests
	@echo "$(BLUE)Running unit tests...$(NC)"
	cd $(BACKEND_DIR) && . venv/bin/activate && pytest tests/ -v -m "not integration and not e2e"

test-integration: setup-ollama ## Run integration tests (requires Ollama)
	@echo "$(BLUE)Running integration tests...$(NC)"
	cd $(BACKEND_DIR) && . venv/bin/activate && pytest tests/ -v -m integration

test-e2e: ## Run end-to-end tests (requires cluster)
	@echo "$(BLUE)Running end-to-end tests...$(NC)"
	@kubectl cluster-info > /dev/null 2>&1 || (echo "$(RED)✗ Kubernetes cluster not accessible$(NC). Run: make cluster-start" && exit 1)
	cd $(BACKEND_DIR) && . venv/bin/activate && pytest tests/ -v -m e2e

test-workflow: setup-ollama ## Run workflow integration test
	@echo "$(BLUE)Running workflow test...$(NC)"
	cd $(BACKEND_DIR) && . venv/bin/activate && $(PYTHON) test_workflow.py

test-watch: ## Run tests in watch mode
	@echo "$(BLUE)Running tests in watch mode...$(NC)"
	cd $(BACKEND_DIR) && . venv/bin/activate && pytest-watch

##@ Code Quality

lint: ## Run linters
	@echo "$(BLUE)Running linters...$(NC)"
	@if [ -d $(BACKEND_VENV) ]; then \
		cd $(BACKEND_DIR) && . venv/bin/activate && \
		(command -v ruff >/dev/null 2>&1 && ruff check src/ ../$(UI_DIR)/*.py || echo "$(YELLOW)ruff not installed, skipping$(NC)"); \
	fi
	@echo "$(GREEN)✓ Linting complete$(NC)"

format: ## Auto-format code
	@echo "$(BLUE)Formatting code...$(NC)"
	@if [ -d $(BACKEND_VENV) ]; then \
		cd $(BACKEND_DIR) && . venv/bin/activate && \
		(command -v black >/dev/null 2>&1 && black src/ ../$(UI_DIR)/*.py || echo "$(YELLOW)black not installed, skipping$(NC)"); \
	fi
	@echo "$(GREEN)✓ Formatting complete$(NC)"

##@ Cleanup

clean: ## Clean generated files and caches
	@echo "$(BLUE)Cleaning generated files...$(NC)"
	rm -rf $(PID_DIR)
	rm -f $(BACKEND_PID).log $(UI_PID).log
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf generated_configs/*.yaml 2>/dev/null || true
	@echo "$(GREEN)✓ Cleanup complete$(NC)"

clean-all: clean ## Clean everything including virtual environments
	@echo "$(BLUE)Cleaning virtual environments...$(NC)"
	rm -rf $(BACKEND_VENV)
	@echo "$(GREEN)✓ Deep cleanup complete$(NC)"

##@ Utilities

info: ## Show configuration and platform info
	@echo "$(BLUE)Platform Information:$(NC)"
	@echo "  Platform: $(PLATFORM)"
	@echo "  OS: $(UNAME_S)"
	@echo "  Arch: $(UNAME_M)"
	@echo "  Python: $(PYTHON) ($$($(PYTHON) --version 2>&1))"
	@echo ""
	@echo "$(BLUE)Configuration:$(NC)"
	@echo "  Registry: $(REGISTRY)"
	@echo "  Org: $(REGISTRY_ORG)"
	@echo "  Simulator Image: $(SIMULATOR_FULL_IMAGE)"
	@echo "  Ollama Model: $(OLLAMA_MODEL)"
	@echo "  KIND Cluster: $(KIND_CLUSTER_NAME)"
	@echo ""
	@echo "$(BLUE)Paths:$(NC)"
	@echo "  Backend: $(BACKEND_DIR)"
	@echo "  UI: $(UI_DIR)"
	@echo "  Simulator: $(SIMULATOR_DIR)"
