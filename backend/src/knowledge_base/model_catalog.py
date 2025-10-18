"""Data access layer for model catalog."""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ModelInfo:
    """Model metadata."""

    def __init__(self, data: dict):
        self.model_id = data["model_id"]
        self.name = data["name"]
        self.provider = data["provider"]
        self.family = data["family"]
        self.size_parameters = data["size_parameters"]
        self.context_length = data["context_length"]
        self.supported_tasks = data["supported_tasks"]
        self.domain_specialization = data["domain_specialization"]
        self.license = data["license"]
        self.license_type = data["license_type"]
        self.min_gpu_memory_gb = data["min_gpu_memory_gb"]
        self.recommended_for = data["recommended_for"]
        self.approval_status = data["approval_status"]
        self.notes = data.get("notes", "")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "name": self.name,
            "provider": self.provider,
            "family": self.family,
            "size_parameters": self.size_parameters,
            "context_length": self.context_length,
            "supported_tasks": self.supported_tasks,
            "domain_specialization": self.domain_specialization,
            "license": self.license,
            "license_type": self.license_type,
            "min_gpu_memory_gb": self.min_gpu_memory_gb,
            "recommended_for": self.recommended_for,
            "approval_status": self.approval_status,
            "notes": self.notes
        }


class GPUType:
    """GPU type metadata."""

    def __init__(self, data: dict):
        self.gpu_type = data["gpu_type"]
        self.memory_gb = data["memory_gb"]
        self.compute_capability = data["compute_capability"]
        self.typical_use_cases = data["typical_use_cases"]
        self.cost_per_hour_usd = data["cost_per_hour_usd"]
        self.availability = data["availability"]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "gpu_type": self.gpu_type,
            "memory_gb": self.memory_gb,
            "compute_capability": self.compute_capability,
            "typical_use_cases": self.typical_use_cases,
            "cost_per_hour_usd": self.cost_per_hour_usd,
            "availability": self.availability
        }


class ModelCatalog:
    """Repository for model and GPU metadata."""

    def __init__(self, data_path: Path | None = None):
        """
        Initialize model catalog.

        Args:
            data_path: Path to model_catalog.json
        """
        if data_path is None:
            data_path = Path(__file__).parent.parent.parent.parent / "data" / "model_catalog.json"

        self.data_path = data_path
        self._models: dict[str, ModelInfo] = {}
        self._gpu_types: dict[str, GPUType] = {}
        self._load_data()

    def _load_data(self):
        """Load catalog data from JSON file."""
        try:
            with open(self.data_path) as f:
                data = json.load(f)

                # Load models
                for model_data in data["models"]:
                    model = ModelInfo(model_data)
                    self._models[model.model_id] = model

                # Load GPU types
                for gpu_data in data["gpu_types"]:
                    gpu = GPUType(gpu_data)
                    self._gpu_types[gpu.gpu_type] = gpu

                logger.info(f"Loaded {len(self._models)} models and {len(self._gpu_types)} GPU types")
        except Exception as e:
            logger.error(f"Failed to load model catalog from {self.data_path}: {e}")
            raise

    def get_model(self, model_id: str) -> ModelInfo | None:
        """Get model by ID."""
        return self._models.get(model_id)

    def get_gpu_type(self, gpu_type: str) -> GPUType | None:
        """Get GPU type metadata."""
        return self._gpu_types.get(gpu_type)

    def find_models_for_use_case(self, use_case: str) -> list[ModelInfo]:
        """
        Find models recommended for a specific use case.

        Args:
            use_case: Use case identifier

        Returns:
            List of recommended models
        """
        return [
            model for model in self._models.values()
            if use_case in model.recommended_for and model.approval_status == "approved"
        ]

    def find_models_by_domain(self, domain: str) -> list[ModelInfo]:
        """
        Find models specialized for a domain.

        Args:
            domain: Domain (e.g., "code", "multilingual")

        Returns:
            List of models with domain specialization
        """
        return [
            model for model in self._models.values()
            if domain in model.domain_specialization and model.approval_status == "approved"
        ]

    def find_models_by_task(self, task: str) -> list[ModelInfo]:
        """
        Find models supporting a specific task.

        Args:
            task: Task type

        Returns:
            List of models supporting this task
        """
        return [
            model for model in self._models.values()
            if task in model.supported_tasks and model.approval_status == "approved"
        ]

    def get_all_models(self) -> list[ModelInfo]:
        """Get all approved models."""
        return [m for m in self._models.values() if m.approval_status == "approved"]

    def get_all_gpu_types(self) -> list[GPUType]:
        """Get all GPU types."""
        return list(self._gpu_types.values())

    def calculate_gpu_cost(
        self,
        gpu_type: str,
        gpu_count: int,
        hours_per_month: float = 730  # ~30 days
    ) -> float | None:
        """
        Calculate monthly GPU cost.

        Args:
            gpu_type: GPU type identifier
            gpu_count: Number of GPUs
            hours_per_month: Hours per month (default: 730)

        Returns:
            Cost in USD, or None if GPU type not found
        """
        gpu = self.get_gpu_type(gpu_type)
        if not gpu:
            return None

        return gpu.cost_per_hour_usd * gpu_count * hours_per_month
