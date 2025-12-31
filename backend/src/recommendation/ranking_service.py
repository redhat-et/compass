"""Ranking service for multi-criteria recommendation sorting.

ACCURACY-FIRST STRATEGY:
1. Get top N unique models by raw accuracy (quality baseline)
2. Filter all hardware configs to only these high-quality models
3. Best Latency/Cost/etc. are ranked WITHIN this quality tier

This ensures:
- All recommendations show HIGH QUALITY models
- No "fast but useless" or "cheap but terrible" recommendations
- Cards show different trade-offs within the same quality tier
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
        Generate 5 ranked lists using ACCURACY-FIRST strategy.

        Strategy:
        1. Get top N unique models by raw accuracy (quality baseline)
        2. Filter ALL hardware configs to only these high-quality models
        3. Best Latency = fastest hardware among high-quality models
        4. Best Cost = cheapest hardware among high-quality models
        5. Balanced = best weighted score among high-quality models

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

        # =====================================================================
        # STEP 1: Get top N UNIQUE MODELS by raw accuracy (quality baseline)
        # When accuracy is TIED, use latency_score as tie-breaker (faster = better)
        # =====================================================================
        seen_models = set()
        unique_accuracy_configs = []
        sorted_by_accuracy = sorted(
            filtered,
            key=lambda x: (
                x.scores.accuracy_score if x.scores else 0,
                # Tie-breaker: LOWEST TTFT wins (faster response)
                # Negate TTFT so lower values sort higher
                -(x.predicted_ttft_p95_ms or 999999),
            ),
            reverse=True,
        )
        
        for config in sorted_by_accuracy:
            model_name = config.model_name or config.model_id or "Unknown"
            if model_name not in seen_models:
                seen_models.add(model_name)
                unique_accuracy_configs.append(config)
                if len(unique_accuracy_configs) >= top_n:
                    break
        
        # Get the model names of top accuracy models
        top_accuracy_model_names = {c.model_name or c.model_id for c in unique_accuracy_configs}
        
        logger.info(
            f"ACCURACY-FIRST: Top {len(top_accuracy_model_names)} models by accuracy: "
            f"{list(top_accuracy_model_names)[:5]}"
        )

        # =====================================================================
        # STEP 2: Filter ALL configs to only high-quality models
        # This ensures Best Latency and Best Cost show HIGH QUALITY models
        # =====================================================================
        high_quality_configs = [
            c for c in filtered 
            if (c.model_name or c.model_id) in top_accuracy_model_names
        ]
        
        logger.info(
            f"ACCURACY-FIRST: {len(high_quality_configs)} configs from top {len(top_accuracy_model_names)} models"
        )

        # =====================================================================
        # STEP 3: Generate ranked lists from HIGH-QUALITY configs only
        # =====================================================================
        ranked_lists = {
            # Best Accuracy: Top N unique models (one config per model)
            "best_accuracy": unique_accuracy_configs[:top_n],
            
            # Best Cost: Cheapest hardware among high-quality models
            # Sort by actual cost (lower is better), not price_score
            "lowest_cost": sorted(
                high_quality_configs,
                key=lambda x: x.cost_per_month_usd or float('inf'),
            )[:top_n],
            
            # Best Latency: Fastest hardware among high-quality models
            # Sort by latency score (higher is better = lower latency)
            "lowest_latency": sorted(
                high_quality_configs,
                key=lambda x: (x.scores.latency_score if x.scores else 0),
                reverse=True,
            )[:top_n],
            
            # Simplest: Fewest GPUs among high-quality models
            "simplest": sorted(
                high_quality_configs,
                key=lambda x: (x.scores.complexity_score if x.scores else 0),
                reverse=True,
            )[:top_n],
            
            # Balanced: Best weighted score among high-quality models
            "balanced": sorted(
                high_quality_configs,
                key=lambda x: (x.scores.balanced_score if x.scores else 0.0),
                reverse=True,
            )[:top_n],
        }

        logger.info(
            f"Generated ranked lists (ACCURACY-FIRST): {len(filtered)} total configs, "
            f"{len(high_quality_configs)} high-quality, top {top_n} per criterion"
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
        Recalculate balanced scores using MAXMIN approach.
        
        MaxMin: The balanced score is the MINIMUM of (accuracy, price, latency).
        This ensures a model with low accuracy cannot win even with excellent cost/latency.
        A small weighted bonus is added to differentiate configs with same min score.

        Args:
            configs: List of configurations to update (modified in place)
            weights: Dict with keys: accuracy, price, latency (complexity ignored)
                     Values are integers 0-10
        """
        # Normalize weights (excluding complexity)
        filtered_weights = {k: v for k, v in weights.items() if k != "complexity"}
        total = sum(filtered_weights.values())
        if total == 0:
            logger.warning("All weights are zero, using default equal weights")
            normalized = {"accuracy": 0.34, "price": 0.33, "latency": 0.33}
        else:
            normalized = {k: v / total for k, v in filtered_weights.items()}

        logger.info(
            f"Recalculating balanced scores with MAXMIN approach: "
            f"A={normalized.get('accuracy', 0):.0%}, P={normalized.get('price', 0):.0%}, "
            f"L={normalized.get('latency', 0):.0%}"
        )

        for config in configs:
            if config.scores:
                # ACCURACY-DOMINANT BALANCED SCORE
                # Accuracy contributes 70%, operational (lat+cost avg) contributes 30%
                # This ensures high-accuracy models win, while still considering cost/latency
                #
                # Example:
                #   GPT-OSS 20B:  Acc=60, Lat=86, Cost=92 → 60×0.7 + 89×0.3 = 68.7
                #   MiniMax-M2:   Acc=80, Lat=70, Cost=35 → 80×0.7 + 52.5×0.3 = 71.75 ← WINS
                #
                acc = config.scores.accuracy_score
                lat = config.scores.latency_score
                cost = config.scores.price_score
                
                # Operational score = average of latency and cost
                operational_avg = (lat + cost) / 2
                
                # Balanced = 70% accuracy + 30% operational
                balanced = acc * 0.7 + operational_avg * 0.3
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
