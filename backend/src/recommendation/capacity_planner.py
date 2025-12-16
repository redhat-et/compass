"""Capacity planning and GPU configuration recommendation.

IMPORTANT: PostgreSQL Migration (Phase 1):
- Uses traffic profile-based exact matching on (prompt_tokens, output_tokens)
- Queries benchmarks by exact traffic profile (512→256, 1024→1024, 4096→512, 10240→1536)
- Filters by p95 SLO compliance (TTFT, ITL, E2E)
- Uses pre-calculated e2e_p95 from benchmarks (not dynamic calculation)

Benchmarks collected using GuideLLM with fixed traffic profiles:
- Batching: vLLM continuous batching (dynamic, auto-configured)
- KV cache: enabled (vLLM default)
- Request pattern: steady-state load

TODO (Phase 2+): Parametric Performance Models
- Train regression models: f(prompt_tokens, output_tokens) → (ttft_p95, itl_p95, e2e_p95)
- Support arbitrary traffic profiles beyond the 4 GuideLLM defaults
- Interpolate for in-range predictions with confidence intervals
"""

import logging
import math

from ..context_intent.schema import (
    ConfigurationScores,
    DeploymentIntent,
    DeploymentRecommendation,
    GPUConfig,
    SLOTargets,
    TrafficProfile,
)
from ..knowledge_base.benchmarks import BenchmarkData, BenchmarkRepository
from ..knowledge_base.model_catalog import ModelCatalog, ModelInfo
from .solution_scorer import SolutionScorer

logger = logging.getLogger(__name__)


class CapacityPlanner:
    """Plan GPU capacity to meet SLO targets and traffic requirements."""

    def __init__(
        self, benchmark_repo: BenchmarkRepository | None = None, catalog: ModelCatalog | None = None
    ):
        """
        Initialize capacity planner.

        Args:
            benchmark_repo: Benchmark repository
            catalog: Model catalog
        """
        self.benchmark_repo = benchmark_repo or BenchmarkRepository()
        self.catalog = catalog or ModelCatalog()

    def plan_capacity(
        self,
        model: ModelInfo,
        traffic_profile: TrafficProfile,
        slo_targets: SLOTargets,
        intent: DeploymentIntent,
    ) -> DeploymentRecommendation | None:
        """
        Plan GPU capacity for a model to meet requirements.

        Uses PostgreSQL to query benchmarks matching exact traffic profile,
        then filters by p95 SLO compliance.

        Args:
            model: Selected model
            traffic_profile: Traffic characteristics (prompt_tokens, output_tokens)
            slo_targets: p95 SLO targets
            intent: Original deployment intent

        Returns:
            DeploymentRecommendation if feasible config found, None otherwise
        """
        # Query PostgreSQL for configurations meeting SLO targets
        # This returns benchmarks that:
        # 1. Match exact traffic profile (prompt_tokens, output_tokens)
        # 2. Meet p95 SLO targets (TTFT, ITL, E2E)
        # 3. Meet minimum QPS requirement
        matching_configs = self.benchmark_repo.find_configurations_meeting_slo(
            prompt_tokens=traffic_profile.prompt_tokens,
            output_tokens=traffic_profile.output_tokens,
            ttft_p95_max_ms=slo_targets.ttft_p95_target_ms,
            itl_p95_max_ms=slo_targets.itl_p95_target_ms,
            e2e_p95_max_ms=slo_targets.e2e_p95_target_ms,
            min_qps=0,  # No minimum QPS filter (we'll scale with replicas)
        )

        if not matching_configs:
            logger.warning(
                f"No configurations found meeting SLO for traffic profile "
                f"({traffic_profile.prompt_tokens}→{traffic_profile.output_tokens})"
            )
            return None

        # Filter to only this model (case-insensitive match)
        model_configs = [
            bench for bench in matching_configs if bench.model_hf_repo.lower() == model.model_id.lower()
        ]

        if not model_configs:
            logger.warning(f"No SLO-compliant configurations found for model {model.model_id}")
            return None

        # Build viable configurations with cost
        viable_configs = []

        for bench in model_configs:
            # Calculate required replicas to handle traffic
            replicas = self._calculate_required_replicas(
                bench.requests_per_second, traffic_profile.expected_qps or 1.0
            )

            # Create GPU config
            gpu_config = GPUConfig(
                gpu_type=bench.hardware,
                gpu_count=bench.hardware_count * replicas,
                tensor_parallel=bench.hardware_count,
                replicas=replicas,
            )

            # Calculate cost
            cost_per_hour = self.catalog.calculate_gpu_cost(
                bench.hardware, gpu_config.gpu_count, hours_per_month=1
            )

            if cost_per_hour is None:
                logger.warning(f"Could not calculate cost for {bench.hardware}")
                continue

            cost_per_month = cost_per_hour * 730  # ~30 days

            # Build recommendation
            recommendation = DeploymentRecommendation(
                intent=intent,
                traffic_profile=traffic_profile,
                slo_targets=slo_targets,
                model_id=model.model_id,
                model_name=model.name,
                gpu_config=gpu_config,
                predicted_ttft_p95_ms=int(bench.ttft_p95) if bench.ttft_p95 else 0,
                predicted_itl_p95_ms=int(bench.itl_p95) if bench.itl_p95 else 0,
                predicted_e2e_p95_ms=int(bench.e2e_p95) if bench.e2e_p95 else 0,
                predicted_throughput_qps=bench.requests_per_second * replicas,
                cost_per_hour_usd=cost_per_hour,
                cost_per_month_usd=cost_per_month,
                meets_slo=True,
                reasoning=self._generate_reasoning(model, bench, gpu_config, intent),
            )

            viable_configs.append((recommendation, cost_per_month))

        if not viable_configs:
            logger.warning(f"No viable configurations found for {model.name}")
            return None

        # Sort by cost
        viable_configs.sort(key=lambda x: x[1])

        # Return best configuration based on budget constraint
        best_recommendation = self._select_best_config(viable_configs, intent.budget_constraint)

        # Add alternative options with full details for tradeoff comparison
        if len(viable_configs) > 1:
            best_recommendation.alternative_options = [
                rec.to_alternative_dict() for rec, _ in viable_configs[1:3]  # Show up to 2 alternatives
            ]

        return best_recommendation

    def _calculate_required_replicas(self, qps_per_replica: float, required_qps: float) -> int:
        """
        Calculate number of replicas needed for traffic.

        Args:
            qps_per_replica: QPS capacity per replica
            required_qps: Required QPS to handle

        Returns:
            Number of replicas (minimum 1)
        """
        # Add 20% headroom for safety
        headroom_factor = 1.2
        required_capacity = required_qps * headroom_factor

        replicas = math.ceil(required_capacity / qps_per_replica)
        return max(1, replicas)

    def _generate_reasoning(
        self,
        model: ModelInfo,
        bench: BenchmarkData,
        gpu_config: GPUConfig,
        intent: DeploymentIntent,
    ) -> str:
        """Generate explanation for recommendation."""
        reasons = []

        # Model selection
        reasons.append(
            f"Selected {model.name} ({model.size_parameters}) for {intent.use_case} use case"
        )

        # GPU configuration
        if gpu_config.tensor_parallel > 1:
            reasons.append(
                f"Using {gpu_config.tensor_parallel}x tensor parallelism on {gpu_config.gpu_type} "
                f"for improved latency"
            )
        else:
            reasons.append(f"Deploying on {gpu_config.gpu_type} GPUs")

        # Scaling
        if gpu_config.replicas > 1:
            reasons.append(
                f"{gpu_config.replicas} independent replicas to handle {intent.user_count} users"
            )

        # Performance
        ttft_p95 = int(bench.ttft_p95) if bench.ttft_p95 else 0
        itl_p95 = int(bench.itl_p95) if bench.itl_p95 else 0
        reasons.append(f"Expected performance: TTFT={ttft_p95}ms (p95), ITL={itl_p95}ms (p95)")

        return ". ".join(reasons)

    def _select_best_config(
        self, configs: list[tuple[DeploymentRecommendation, float]], budget_constraint: str
    ) -> DeploymentRecommendation:
        """
        Select best configuration based on budget constraint.

        Args:
            configs: List of (recommendation, cost) tuples, sorted by cost
            budget_constraint: Budget sensitivity

        Returns:
            Best recommendation
        """
        if budget_constraint in ["strict", "moderate"]:
            # Return cheapest option
            return configs[0][0]
        elif budget_constraint == "flexible":
            # Return middle option if available
            mid_idx = len(configs) // 2
            return configs[mid_idx][0]
        else:  # "none"
            # Return most performant (likely most expensive)
            # Sort by predicted latency instead of cost
            configs_by_perf = sorted(
                configs, key=lambda x: (x[0].predicted_ttft_p95_ms + x[0].predicted_itl_p95_ms)
            )
            return configs_by_perf[0][0]

    def plan_all_capacities(
        self,
        models: list[ModelInfo],
        traffic_profile: TrafficProfile,
        slo_targets: SLOTargets,
        intent: DeploymentIntent,
        include_near_miss: bool = True,
        near_miss_tolerance: float = 0.2,
    ) -> list[DeploymentRecommendation]:
        """
        Plan GPU capacity for multiple models and return ALL viable configurations.

        Unlike plan_capacity(), this returns all configurations that meet or nearly
        meet SLO targets, with full scoring attached for multi-criteria ranking.

        Args:
            models: List of models to evaluate
            traffic_profile: Traffic characteristics (prompt_tokens, output_tokens)
            slo_targets: p95 SLO targets
            intent: Original deployment intent
            include_near_miss: Whether to include configs within tolerance of SLO
            near_miss_tolerance: How much over SLO to allow (0.2 = 20%)

        Returns:
            List of DeploymentRecommendations with scores attached
        """
        scorer = SolutionScorer()
        all_configs: list[DeploymentRecommendation] = []

        # Determine SLO thresholds for query
        # If including near-miss, relax thresholds by tolerance
        if include_near_miss:
            query_ttft = int(slo_targets.ttft_p95_target_ms * (1 + near_miss_tolerance))
            query_itl = int(slo_targets.itl_p95_target_ms * (1 + near_miss_tolerance))
            query_e2e = int(slo_targets.e2e_p95_target_ms * (1 + near_miss_tolerance))
        else:
            query_ttft = slo_targets.ttft_p95_target_ms
            query_itl = slo_targets.itl_p95_target_ms
            query_e2e = slo_targets.e2e_p95_target_ms

        # Query PostgreSQL for configurations meeting relaxed SLO targets
        matching_configs = self.benchmark_repo.find_configurations_meeting_slo(
            prompt_tokens=traffic_profile.prompt_tokens,
            output_tokens=traffic_profile.output_tokens,
            ttft_p95_max_ms=query_ttft,
            itl_p95_max_ms=query_itl,
            e2e_p95_max_ms=query_e2e,
            min_qps=0,
        )

        if not matching_configs:
            logger.warning(
                f"No configurations found for traffic profile "
                f"({traffic_profile.prompt_tokens}→{traffic_profile.output_tokens})"
            )
            return []

        # Build model lookup for quick access
        model_lookup = {m.model_id.lower(): m for m in models}

        # Process each matching benchmark
        for bench in matching_configs:
            # Check if this model is in our candidate list
            model = model_lookup.get(bench.model_hf_repo.lower())
            if not model:
                continue

            # Calculate required replicas to handle traffic
            replicas = self._calculate_required_replicas(
                bench.requests_per_second, traffic_profile.expected_qps or 1.0
            )

            # Create GPU config
            gpu_config = GPUConfig(
                gpu_type=bench.hardware,
                gpu_count=bench.hardware_count * replicas,
                tensor_parallel=bench.hardware_count,
                replicas=replicas,
            )

            # Calculate cost
            cost_per_hour = self.catalog.calculate_gpu_cost(
                bench.hardware, gpu_config.gpu_count, hours_per_month=1
            )

            if cost_per_hour is None:
                logger.warning(f"Could not calculate cost for {bench.hardware}")
                continue

            cost_per_month = cost_per_hour * 730  # ~30 days

            # Calculate latency score and SLO status
            predicted_ttft = int(bench.ttft_p95) if bench.ttft_p95 else 0
            predicted_itl = int(bench.itl_p95) if bench.itl_p95 else 0
            predicted_e2e = int(bench.e2e_p95) if bench.e2e_p95 else 0

            latency_score, slo_status = scorer.score_latency(
                predicted_ttft_ms=predicted_ttft,
                predicted_itl_ms=predicted_itl,
                predicted_e2e_ms=predicted_e2e,
                target_ttft_ms=slo_targets.ttft_p95_target_ms,
                target_itl_ms=slo_targets.itl_p95_target_ms,
                target_e2e_ms=slo_targets.e2e_p95_target_ms,
            )

            # Skip if exceeds SLO and we're not including near-miss
            if slo_status == "exceeds" and not include_near_miss:
                continue

            # Calculate other scores
            accuracy_score = scorer.score_accuracy(model.size_parameters)
            complexity_score = scorer.score_complexity(gpu_config.gpu_count)

            # Build recommendation (price score calculated later after we know min/max)
            recommendation = DeploymentRecommendation(
                intent=intent,
                traffic_profile=traffic_profile,
                slo_targets=slo_targets,
                model_id=model.model_id,
                model_name=model.name,
                gpu_config=gpu_config,
                predicted_ttft_p95_ms=predicted_ttft,
                predicted_itl_p95_ms=predicted_itl,
                predicted_e2e_p95_ms=predicted_e2e,
                predicted_throughput_qps=bench.requests_per_second * replicas,
                cost_per_hour_usd=cost_per_hour,
                cost_per_month_usd=cost_per_month,
                meets_slo=(slo_status == "compliant"),
                reasoning=self._generate_reasoning(model, bench, gpu_config, intent),
                # Temporary scores without price (will be updated below)
                scores=ConfigurationScores(
                    accuracy_score=accuracy_score,
                    price_score=0,  # Placeholder
                    latency_score=latency_score,
                    complexity_score=complexity_score,
                    balanced_score=0.0,  # Placeholder
                    slo_status=slo_status,
                ),
            )

            all_configs.append(recommendation)

        if not all_configs:
            logger.warning("No viable configurations found for any model")
            return []

        # Now calculate price scores (need min/max across all configs)
        costs = [rec.cost_per_month_usd for rec in all_configs if rec.cost_per_month_usd]
        if costs:
            min_cost = min(costs)
            max_cost = max(costs)

            for rec in all_configs:
                if rec.scores and rec.cost_per_month_usd:
                    # Update price score
                    price_score = scorer.score_price(
                        rec.cost_per_month_usd, min_cost, max_cost
                    )
                    rec.scores.price_score = price_score

                    # Calculate balanced score
                    rec.scores.balanced_score = scorer.score_balanced(
                        accuracy_score=rec.scores.accuracy_score,
                        price_score=price_score,
                        latency_score=rec.scores.latency_score,
                        complexity_score=rec.scores.complexity_score,
                    )

        logger.info(f"Found {len(all_configs)} viable configurations across {len(models)} models")
        return all_configs
