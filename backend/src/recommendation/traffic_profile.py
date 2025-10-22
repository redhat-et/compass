"""Traffic profile generation from deployment intent."""

import logging

from ..context_intent.schema import DeploymentIntent, SLOTargets, TrafficProfile
from ..knowledge_base.slo_templates import SLOTemplateRepository

logger = logging.getLogger(__name__)


class TrafficProfileGenerator:
    """Generate traffic profiles and SLO targets from deployment intent."""

    def __init__(self, slo_repo: SLOTemplateRepository | None = None):
        """
        Initialize traffic profile generator.

        Args:
            slo_repo: SLO template repository (creates default if not provided)
        """
        self.slo_repo = slo_repo or SLOTemplateRepository()

    def generate_profile(self, intent: DeploymentIntent) -> TrafficProfile:
        """
        Generate traffic profile from deployment intent.

        Args:
            intent: Deployment intent

        Returns:
            TrafficProfile with estimated characteristics
        """
        # Get base template for use case
        template = self.slo_repo.get_template(intent.use_case)

        if not template:
            logger.warning(f"No template found for use_case={intent.use_case}, using defaults")
            return self._generate_default_profile(intent)

        # Calculate expected QPS based on user count and request frequency
        expected_qps = self._estimate_qps(
            user_count=intent.user_count,
            requests_per_user_per_day=template.requests_per_user_per_day or 10,
            latency_requirement=intent.latency_requirement,
        )

        return TrafficProfile(
            prompt_tokens_mean=template.prompt_tokens_mean,
            prompt_tokens_variance=template.prompt_tokens_variance,
            generation_tokens_mean=template.generation_tokens_mean,
            generation_tokens_variance=template.generation_tokens_variance,
            expected_qps=expected_qps,
            requests_per_user_per_day=template.requests_per_user_per_day,
        )

    def generate_slo_targets(self, intent: DeploymentIntent) -> SLOTargets:
        """
        Generate SLO targets from deployment intent.

        Args:
            intent: Deployment intent

        Returns:
            SLOTargets with target latencies
        """
        # Get base template for use case
        template = self.slo_repo.get_template(intent.use_case)

        if not template:
            logger.warning(f"No template found for use_case={intent.use_case}, using defaults")
            return self._generate_default_slo(intent)

        # Adjust SLO targets based on latency requirement
        ttft_target = self._adjust_slo_for_latency(
            template.ttft_p90_target_ms, intent.latency_requirement
        )
        tpot_target = self._adjust_slo_for_latency(
            template.tpot_p90_target_ms, intent.latency_requirement
        )
        e2e_target = self._adjust_slo_for_latency(
            template.e2e_p90_target_ms, intent.latency_requirement
        )

        return SLOTargets(
            ttft_p90_target_ms=ttft_target,
            tpot_p90_target_ms=tpot_target,
            e2e_p90_target_ms=e2e_target,
        )

    def _estimate_qps(
        self, user_count: int, requests_per_user_per_day: int, latency_requirement: str
    ) -> float:
        """
        Estimate peak QPS based on user count and usage patterns.

        Args:
            user_count: Number of users
            requests_per_user_per_day: Average requests per user per day
            latency_requirement: Latency sensitivity

        Returns:
            Estimated peak QPS
        """
        # Total requests per day
        total_requests_per_day = user_count * requests_per_user_per_day

        # Assume traffic is concentrated in peak hours (e.g., 8 hours)
        # with a peak-to-average ratio
        peak_hours = 8
        peak_ratio_map = {
            "very_high": 3.0,  # High variability, need headroom
            "high": 2.5,
            "medium": 2.0,
            "low": 1.5,
        }
        peak_ratio = peak_ratio_map.get(latency_requirement, 2.0)

        # Calculate average QPS during peak hours
        avg_qps_peak = total_requests_per_day / (peak_hours * 3600)

        # Apply peak ratio
        peak_qps = avg_qps_peak * peak_ratio

        # Ensure minimum QPS of 0.1 for small workloads
        peak_qps = max(0.1, peak_qps)

        return round(peak_qps, 2)

    def _adjust_slo_for_latency(self, base_target_ms: int, latency_requirement: str) -> int:
        """
        Adjust SLO target based on user's latency requirement.

        Args:
            base_target_ms: Base SLO target from template
            latency_requirement: User's latency sensitivity

        Returns:
            Adjusted SLO target
        """
        adjustment_factors = {
            "very_high": 0.9,  # 10% tighter than template (more realistic)
            "high": 0.95,  # 5% tighter
            "medium": 1.0,  # Use template as-is
            "low": 1.2,  # 20% more relaxed
        }

        factor = adjustment_factors.get(latency_requirement, 1.0)
        return int(base_target_ms * factor)

    def _generate_default_profile(self, intent: DeploymentIntent) -> TrafficProfile:
        """Generate default profile when no template available."""
        # Conservative defaults
        default_qps = intent.user_count * 0.01  # 1% of users active concurrently

        return TrafficProfile(
            prompt_tokens_mean=300,
            prompt_tokens_variance=150,
            generation_tokens_mean=200,
            generation_tokens_variance=100,
            expected_qps=default_qps,
            requests_per_user_per_day=10,
        )

    def _generate_default_slo(self, intent: DeploymentIntent) -> SLOTargets:
        """Generate default SLO targets when no template available."""
        slo_map = {
            "very_high": (150, 40, 1500),
            "high": (250, 60, 3000),
            "medium": (500, 80, 5000),
            "low": (1000, 100, 10000),
        }

        ttft, tpot, e2e = slo_map.get(intent.latency_requirement, (500, 80, 5000))

        return SLOTargets(ttft_p90_target_ms=ttft, tpot_p90_target_ms=tpot, e2e_p90_target_ms=e2e)
