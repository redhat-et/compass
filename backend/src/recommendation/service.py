"""Recommendation Service facade.

Provides a high-level interface for generating deployment recommendations.
"""

import logging

from ..shared.schemas import (
    DeploymentIntent,
    DeploymentRecommendation,
    RankedRecommendationsResponse,
    SLOTargets,
    TrafficProfile,
)
from .config_finder import ConfigFinder
from .analyzer import Analyzer

logger = logging.getLogger(__name__)


class RecommendationService:
    """High-level service for generating deployment recommendations."""

    def __init__(self):
        """Initialize the Recommendation Service."""
        self.config_finder = ConfigFinder()
        self.analyzer = Analyzer()

    def generate_recommendation(
        self,
        intent: DeploymentIntent,
        traffic_profile: TrafficProfile,
        slo_targets: SLOTargets,
    ) -> DeploymentRecommendation:
        """
        Generate a single best recommendation.

        Args:
            intent: Deployment intent
            traffic_profile: Traffic profile with token counts and QPS
            slo_targets: SLO targets to meet

        Returns:
            Best DeploymentRecommendation

        Raises:
            ValueError: If no viable configuration found
        """
        # Get all viable configurations
        all_configs = self.config_finder.plan_all_capacities(
            traffic_profile=traffic_profile,
            slo_targets=slo_targets,
            intent=intent,
            include_near_miss=False,
        )

        if not all_configs:
            raise ValueError(
                f"No viable configurations found for use_case={intent.use_case} "
                f"meeting SLO targets"
            )

        # Sort by balanced score and pick best
        all_configs.sort(
            key=lambda x: x.scores.balanced_score if x.scores else 0,
            reverse=True,
        )
        best = all_configs[0]

        # Add alternatives
        if len(all_configs) > 1:
            best.alternative_options = [
                rec.to_alternative_dict() for rec in all_configs[1:4]
            ]

        logger.info(
            f"Generated recommendation: {best.model_name} on "
            f"{best.gpu_config.gpu_count}x {best.gpu_config.gpu_type}"
        )

        return best

    def generate_ranked_recommendations(
        self,
        intent: DeploymentIntent,
        traffic_profile: TrafficProfile,
        slo_targets: SLOTargets,
        min_accuracy: int | None = None,
        max_cost: float | None = None,
        include_near_miss: bool = True,
        weights: dict[str, int] | None = None,
    ) -> RankedRecommendationsResponse:
        """
        Generate multiple ranked recommendation lists.

        Returns 5 different views of the same configurations,
        each sorted by a different criterion.

        Args:
            intent: Deployment intent
            traffic_profile: Traffic profile
            slo_targets: SLO targets
            min_accuracy: Minimum accuracy score filter (0-100)
            max_cost: Maximum monthly cost filter (USD)
            include_near_miss: Whether to include near-SLO configurations
            weights: Custom weights for balanced score

        Returns:
            RankedRecommendationsResponse with 5 ranked lists
        """
        from ..shared.schemas import DeploymentSpecification

        # Get all viable configurations
        all_configs = self.config_finder.plan_all_capacities(
            traffic_profile=traffic_profile,
            slo_targets=slo_targets,
            intent=intent,
            include_near_miss=include_near_miss,
            weights=weights,
        )

        specification = DeploymentSpecification(
            intent=intent,
            traffic_profile=traffic_profile,
            slo_targets=slo_targets,
        )

        if not all_configs:
            logger.warning("No viable configurations found")
            return RankedRecommendationsResponse(
                min_accuracy_threshold=min_accuracy,
                max_cost_ceiling=max_cost,
                include_near_miss=include_near_miss,
                specification=specification,
                total_configs_evaluated=0,
                configs_after_filters=0,
            )

        # Generate ranked lists
        ranked_lists = self.analyzer.generate_ranked_lists(
            configurations=all_configs,
            min_accuracy=min_accuracy,
            max_cost=max_cost,
            top_n=10,
            weights=weights,
            use_case=intent.use_case,
        )

        configs_after_filters = self.analyzer.get_unique_configs_count(
            ranked_lists
        )

        logger.info(
            f"Generated ranked recommendations: {len(all_configs)} configs, "
            f"{configs_after_filters} after filters"
        )

        return RankedRecommendationsResponse(
            min_accuracy_threshold=min_accuracy,
            max_cost_ceiling=max_cost,
            include_near_miss=include_near_miss,
            specification=specification,
            best_accuracy=ranked_lists["best_accuracy"],
            lowest_cost=ranked_lists["lowest_cost"],
            lowest_latency=ranked_lists["lowest_latency"],
            simplest=ranked_lists["simplest"],
            balanced=ranked_lists["balanced"],
            total_configs_evaluated=len(all_configs),
            configs_after_filters=configs_after_filters,
        )
