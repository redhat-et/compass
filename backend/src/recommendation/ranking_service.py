"""Ranking service for multi-criteria recommendation sorting.

Generates 5 ranked lists from deployment configurations:
- Best Accuracy: Sorted by model capability score
- Lowest Cost: Sorted by price efficiency score
- Lowest Latency: Sorted by SLO headroom score
- Simplest: Sorted by deployment complexity score
- Balanced: Sorted by weighted composite score
"""

import logging

from ..context_intent.schema import DeploymentRecommendation

logger = logging.getLogger(__name__)


class RankingService:
    """Generate ranked recommendation lists from scored configurations."""

    def generate_ranked_lists(
        self,
        configurations: list[DeploymentRecommendation],
        min_accuracy: int | None = None,
        max_cost: float | None = None,
        top_n: int = 5,
        weights: dict[str, int] | None = None,
    ) -> dict[str, list[DeploymentRecommendation]]:
        """
        Generate 5 ranked lists from configurations.

        Args:
            configurations: List of scored DeploymentRecommendations
            min_accuracy: Minimum accuracy score filter (0-100)
            max_cost: Maximum monthly cost filter (USD)
            top_n: Number of top configurations to return per list
            weights: Optional custom weights for balanced score (0-10 scale)
                     Keys: accuracy, price, latency, complexity

        Returns:
            Dict with keys: best_accuracy, lowest_cost, lowest_latency,
                           simplest, balanced
        """
        # Apply filters
        filtered = self._apply_filters(configurations, min_accuracy, max_cost)

        # Recalculate balanced scores if custom weights provided
        if weights and filtered:
            self._recalculate_balanced_scores(filtered, weights)

        if not filtered:
            logger.warning("No configurations remain after filtering")
            return {
                "best_accuracy": [],
                "lowest_cost": [],
                "lowest_latency": [],
                "simplest": [],
                "balanced": [],
            }

        # Generate sorted lists
        # Higher score is better for all criteria
        ranked_lists = {
            "best_accuracy": sorted(
                filtered,
                key=lambda x: (x.scores.accuracy_score if x.scores else 0),
                reverse=True,
            )[:top_n],
            "lowest_cost": sorted(
                filtered,
                key=lambda x: (x.scores.price_score if x.scores else 0),
                reverse=True,
            )[:top_n],
            "lowest_latency": sorted(
                filtered,
                key=lambda x: (x.scores.latency_score if x.scores else 0),
                reverse=True,
            )[:top_n],
            "simplest": sorted(
                filtered,
                key=lambda x: (x.scores.complexity_score if x.scores else 0),
                reverse=True,
            )[:top_n],
            "balanced": sorted(
                filtered,
                key=lambda x: (x.scores.balanced_score if x.scores else 0.0),
                reverse=True,
            )[:top_n],
        }

        logger.info(
            f"Generated ranked lists: {len(filtered)} configs after filters, "
            f"top {top_n} per criterion"
        )

        return ranked_lists

    def _apply_filters(
        self,
        configs: list[DeploymentRecommendation],
        min_accuracy: int | None,
        max_cost: float | None,
    ) -> list[DeploymentRecommendation]:
        """
        Apply accuracy and cost filters to configurations.

        Args:
            configs: List of configurations to filter
            min_accuracy: Minimum accuracy score (0-100), None = no filter
            max_cost: Maximum monthly cost (USD), None = no filter

        Returns:
            Filtered list of configurations
        """
        filtered = configs

        # Filter by minimum accuracy
        if min_accuracy is not None and min_accuracy > 0:
            filtered = [
                c for c in filtered
                if c.scores and c.scores.accuracy_score >= min_accuracy
            ]
            logger.debug(f"After min_accuracy={min_accuracy} filter: {len(filtered)} configs")

        # Filter by maximum cost
        if max_cost is not None and max_cost > 0:
            filtered = [
                c for c in filtered
                if c.cost_per_month_usd is not None and c.cost_per_month_usd <= max_cost
            ]
            logger.debug(f"After max_cost=${max_cost} filter: {len(filtered)} configs")

        return filtered

    def _recalculate_balanced_scores(
        self,
        configs: list[DeploymentRecommendation],
        weights: dict[str, int],
    ) -> None:
        """
        Recalculate balanced scores using custom weights.

        The weights are provided on a 0-10 scale and are normalized
        to percentages that sum to 1.0.

        Args:
            configs: List of configurations to update (modified in place)
            weights: Dict with keys: accuracy, price, latency, complexity
                     Values are integers 0-10
        """
        # Normalize weights to percentages
        total = sum(weights.values())
        if total == 0:
            logger.warning("All weights are zero, using default equal weights")
            normalized = {"accuracy": 0.25, "price": 0.25, "latency": 0.25, "complexity": 0.25}
        else:
            normalized = {k: v / total for k, v in weights.items()}

        logger.info(
            f"Recalculating balanced scores with weights: "
            f"A={normalized['accuracy']:.0%}, P={normalized['price']:.0%}, "
            f"L={normalized['latency']:.0%}, C={normalized['complexity']:.0%}"
        )

        for config in configs:
            if config.scores:
                balanced = (
                    config.scores.accuracy_score * normalized["accuracy"]
                    + config.scores.price_score * normalized["price"]
                    + config.scores.latency_score * normalized["latency"]
                    + config.scores.complexity_score * normalized["complexity"]
                )
                config.scores.balanced_score = round(balanced, 1)

    def get_unique_configs_count(
        self, ranked_lists: dict[str, list[DeploymentRecommendation]]
    ) -> int:
        """
        Count unique configurations across all ranked lists.

        Since the same configuration may appear in multiple lists
        (e.g., best accuracy AND lowest cost), this counts unique ones.

        Args:
            ranked_lists: Dict of ranked lists

        Returns:
            Count of unique configurations
        """
        seen = set()
        for configs in ranked_lists.values():
            for config in configs:
                # Use model_id + gpu_config as unique key
                if config.gpu_config:
                    key = (
                        config.model_id,
                        config.gpu_config.gpu_type,
                        config.gpu_config.gpu_count,
                        config.gpu_config.tensor_parallel,
                        config.gpu_config.replicas,
                    )
                    seen.add(key)
        return len(seen)
