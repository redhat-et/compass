"""Specification Service facade.

Provides a high-level interface for generating deployment specifications.
"""

import logging

from ..shared.schemas import (
    DeploymentIntent,
    DeploymentSpecification,
    SLOTargets,
    TrafficProfile,
)
from ..knowledge_base.slo_templates import SLOTemplateRepository
from .traffic_profile import TrafficProfileGenerator

logger = logging.getLogger(__name__)


class SpecificationService:
    """High-level service for generating deployment specifications from intent."""

    def __init__(self, slo_repo: SLOTemplateRepository | None = None):
        """
        Initialize the Specification Service.

        Args:
            slo_repo: Optional SLO template repository
        """
        self.traffic_generator = TrafficProfileGenerator(slo_repo)
        self.slo_repo = slo_repo or SLOTemplateRepository()

    def generate_specification(
        self, intent: DeploymentIntent
    ) -> DeploymentSpecification:
        """
        Generate a complete deployment specification from intent.

        This is the primary method for obtaining a deployment specification
        from a validated DeploymentIntent.

        Args:
            intent: The deployment intent

        Returns:
            Complete DeploymentSpecification with traffic profile and SLO targets
        """
        logger.info(f"Generating specification for use_case={intent.use_case}")

        # Generate traffic profile
        traffic_profile = self.traffic_generator.generate_profile(intent)
        logger.info(
            f"Traffic profile: {traffic_profile.prompt_tokens}→{traffic_profile.output_tokens} tokens, "
            f"{traffic_profile.expected_qps} QPS"
        )

        # Generate SLO targets
        slo_targets = self.traffic_generator.generate_slo_targets(intent)
        logger.info(
            f"SLO targets (p95): TTFT={slo_targets.ttft_p95_target_ms}ms, "
            f"ITL={slo_targets.itl_p95_target_ms}ms, E2E={slo_targets.e2e_p95_target_ms}ms"
        )

        return DeploymentSpecification(
            intent=intent,
            traffic_profile=traffic_profile,
            slo_targets=slo_targets,
        )

    def generate_traffic_profile(self, intent: DeploymentIntent) -> TrafficProfile:
        """
        Generate traffic profile from intent.

        Args:
            intent: The deployment intent

        Returns:
            TrafficProfile with token counts and expected QPS
        """
        return self.traffic_generator.generate_profile(intent)

    def generate_slo_targets(self, intent: DeploymentIntent) -> SLOTargets:
        """
        Generate SLO targets from intent.

        Args:
            intent: The deployment intent

        Returns:
            SLOTargets with p95 latency targets
        """
        return self.traffic_generator.generate_slo_targets(intent)
