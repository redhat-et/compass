"""Data access layer for use case SLO templates."""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, List

logger = logging.getLogger(__name__)


class SLOTemplate:
    """SLO template for a specific use case."""

    def __init__(self, use_case: str, data: dict):
        self.use_case = use_case
        self.name = data["name"]
        self.description = data["description"]

        # SLO targets
        slo = data["slo_targets"]
        self.ttft_p90_target_ms = slo["ttft_p90_target_ms"]
        self.tpot_p90_target_ms = slo["tpot_p90_target_ms"]
        self.e2e_p95_target_ms = slo["e2e_p95_target_ms"]

        # Typical traffic characteristics
        traffic = data["typical_traffic"]
        self.prompt_tokens_mean = traffic["prompt_tokens_mean"]
        self.prompt_tokens_variance = traffic.get("prompt_tokens_variance")
        self.generation_tokens_mean = traffic["generation_tokens_mean"]
        self.generation_tokens_variance = traffic.get("generation_tokens_variance")
        self.requests_per_user_per_day = traffic.get("requests_per_user_per_day")

        # Business context
        context = data["business_context"]
        self.user_facing = context["user_facing"]
        self.latency_sensitivity = context["latency_sensitivity"]
        self.throughput_priority = context["throughput_priority"]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "use_case": self.use_case,
            "name": self.name,
            "description": self.description,
            "slo_targets": {
                "ttft_p90_target_ms": self.ttft_p90_target_ms,
                "tpot_p90_target_ms": self.tpot_p90_target_ms,
                "e2e_p95_target_ms": self.e2e_p95_target_ms
            },
            "typical_traffic": {
                "prompt_tokens_mean": self.prompt_tokens_mean,
                "prompt_tokens_variance": self.prompt_tokens_variance,
                "generation_tokens_mean": self.generation_tokens_mean,
                "generation_tokens_variance": self.generation_tokens_variance,
                "requests_per_user_per_day": self.requests_per_user_per_day
            },
            "business_context": {
                "user_facing": self.user_facing,
                "latency_sensitivity": self.latency_sensitivity,
                "throughput_priority": self.throughput_priority
            }
        }


class SLOTemplateRepository:
    """Repository for use case SLO templates."""

    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize SLO template repository.

        Args:
            data_path: Path to slo_templates.json
        """
        if data_path is None:
            data_path = Path(__file__).parent.parent.parent.parent / "data" / "slo_templates.json"

        self.data_path = data_path
        self._templates: Dict[str, SLOTemplate] = {}
        self._load_data()

    def _load_data(self):
        """Load SLO templates from JSON file."""
        try:
            with open(self.data_path) as f:
                data = json.load(f)
                for use_case, template_data in data["use_cases"].items():
                    self._templates[use_case] = SLOTemplate(use_case, template_data)
                logger.info(f"Loaded {len(self._templates)} SLO templates")
        except Exception as e:
            logger.error(f"Failed to load SLO templates from {self.data_path}: {e}")
            raise

    def get_template(self, use_case: str) -> Optional[SLOTemplate]:
        """
        Get SLO template for a specific use case.

        Args:
            use_case: Use case identifier

        Returns:
            SLOTemplate if found, None otherwise
        """
        return self._templates.get(use_case)

    def get_all_templates(self) -> Dict[str, SLOTemplate]:
        """Get all SLO templates."""
        return self._templates.copy()

    def list_use_cases(self) -> List[str]:
        """Get list of all supported use cases."""
        return list(self._templates.keys())
