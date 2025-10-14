"""Capacity planning and GPU configuration recommendation.

IMPORTANT LIMITATIONS (Phase 1 POC):
Current benchmark data uses point estimates assuming typical conditions:
- Input tokens: ~150 (average)
- Output tokens: ~200 (average)
- Request pattern: steady (non-bursty)
- Batching: vLLM continuous batching (dynamic, auto-configured)
- KV cache: enabled (vLLM default)

In reality, performance varies significantly with:
1. Input/output token lengths (longer → higher latency, lower throughput)
2. Concurrency levels and request arrival patterns (bursty traffic degrades tail latencies)
3. KV cache efficiency (prefix caching effectiveness depends on prompt similarity)
4. vLLM configuration tuning for specific workload patterns

TODO (Phase 2): Implement multi-dimensional benchmark lookups
- Store benchmarks for varying input/output token distributions
- Interpolate between benchmark points for non-exact matches
- Model queueing effects and concurrency impact on latency
- Account for traffic burstiness and KV cache hit rates
- Consider parametric models (regression-based) for continuous prediction

See data/benchmarks.json _metadata for detailed benchmark conditions.
"""

import logging
import math
from typing import Optional, List, Tuple

from ..context_intent.schema import (
    DeploymentIntent,
    TrafficProfile,
    SLOTargets,
    GPUConfig,
    DeploymentRecommendation
)
from ..knowledge_base.benchmarks import BenchmarkRepository, BenchmarkData
from ..knowledge_base.model_catalog import ModelCatalog, ModelInfo

logger = logging.getLogger(__name__)


class CapacityPlanner:
    """Plan GPU capacity to meet SLO targets and traffic requirements."""

    def __init__(
        self,
        benchmark_repo: Optional[BenchmarkRepository] = None,
        catalog: Optional[ModelCatalog] = None
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
        intent: DeploymentIntent
    ) -> Optional[DeploymentRecommendation]:
        """
        Plan GPU capacity for a model to meet requirements.

        Args:
            model: Selected model
            traffic_profile: Traffic characteristics
            slo_targets: SLO targets
            intent: Original deployment intent

        Returns:
            DeploymentRecommendation if feasible config found, None otherwise
        """
        # Get all benchmarks for this model
        model_benchmarks = self.benchmark_repo.get_benchmarks_for_model(model.model_id)

        if not model_benchmarks:
            logger.warning(f"No benchmarks found for model {model.model_id}")
            return None

        # Find configurations that meet SLO targets
        viable_configs = []

        for bench in model_benchmarks:
            # Check if this benchmark meets SLO requirements (TTFT and TPOT)
            if not self._meets_slo_targets(bench, slo_targets):
                continue

            # Calculate E2E latency estimate
            predicted_e2e = self._estimate_e2e_latency(bench, traffic_profile)

            # Check if E2E also meets target
            if predicted_e2e > slo_targets.e2e_p95_target_ms:
                logger.debug(
                    f"Skipping {bench.gpu_type} TP={bench.tensor_parallel}: "
                    f"E2E {predicted_e2e}ms > target {slo_targets.e2e_p95_target_ms}ms"
                )
                continue

            # Calculate required replicas to handle traffic
            replicas = self._calculate_required_replicas(
                bench.max_qps,
                traffic_profile.expected_qps
            )

            # Create GPU config
            gpu_config = GPUConfig(
                gpu_type=bench.gpu_type,
                gpu_count=bench.tensor_parallel * replicas,
                tensor_parallel=bench.tensor_parallel,
                replicas=replicas
            )

            # Calculate cost
            cost_per_hour = self.catalog.calculate_gpu_cost(
                bench.gpu_type,
                gpu_config.gpu_count,
                hours_per_month=1
            )

            if cost_per_hour is None:
                logger.warning(f"Could not calculate cost for {bench.gpu_type}")
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
                predicted_ttft_p90_ms=bench.ttft_p90_ms,
                predicted_tpot_p90_ms=bench.tpot_p90_ms,
                predicted_e2e_p95_ms=predicted_e2e,
                predicted_throughput_qps=bench.max_qps * replicas,
                cost_per_hour_usd=cost_per_hour,
                cost_per_month_usd=cost_per_month,
                meets_slo=True,
                reasoning=self._generate_reasoning(model, bench, gpu_config, intent)
            )

            viable_configs.append((recommendation, cost_per_month))

        if not viable_configs:
            logger.warning(f"No viable configurations found for {model.name}")
            return None

        # Sort by cost and budget constraint
        viable_configs.sort(key=lambda x: x[1])

        # Return best configuration based on budget constraint
        best_recommendation = self._select_best_config(
            viable_configs,
            intent.budget_constraint
        )

        # Add alternative options with full details for tradeoff comparison
        if len(viable_configs) > 1:
            best_recommendation.alternative_options = [
                {
                    "model_name": rec.model_name,
                    "model_id": rec.model_id,
                    "gpu_config": rec.gpu_config.dict(),
                    "predicted_ttft_p90_ms": rec.predicted_ttft_p90_ms,
                    "predicted_tpot_p90_ms": rec.predicted_tpot_p90_ms,
                    "predicted_e2e_p95_ms": rec.predicted_e2e_p95_ms,
                    "predicted_throughput_qps": rec.predicted_throughput_qps,
                    "cost_per_hour_usd": rec.cost_per_hour_usd,
                    "cost_per_month_usd": rec.cost_per_month_usd,
                    "reasoning": rec.reasoning
                }
                for rec, _ in viable_configs[1:3]  # Show up to 2 alternatives
            ]

        return best_recommendation

    def _meets_slo_targets(self, bench: BenchmarkData, slo: SLOTargets) -> bool:
        """Check if benchmark meets SLO targets."""
        return (
            bench.ttft_p90_ms <= slo.ttft_p90_target_ms and
            bench.tpot_p90_ms <= slo.tpot_p90_target_ms
        )

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

    def _estimate_e2e_latency(
        self,
        bench: BenchmarkData,
        traffic_profile: TrafficProfile
    ) -> int:
        """
        Estimate end-to-end latency for streaming inference.

        For streaming mode (typical for LLM deployments), E2E latency is dominated
        by TTFT and a small number of initial tokens, not the full generation.
        Users perceive latency as "time until useful response starts appearing".

        Args:
            bench: Benchmark data
            traffic_profile: Traffic characteristics

        Returns:
            Estimated E2E p95 latency (ms)
        """
        # For streaming: E2E ≈ TTFT + (first ~20 tokens × TPOT)
        # This represents the time until the user has a meaningful response
        ttft = bench.ttft_p90_ms
        tpot = bench.tpot_p90_ms

        # Use first 20 tokens as "perceived latency"
        # (users typically start reading after this much content)
        perceived_gen_tokens = min(20, traffic_profile.generation_tokens_mean)

        e2e_p90 = ttft + (tpot * perceived_gen_tokens)

        # Add ~20% buffer for p95
        e2e_p95 = int(e2e_p90 * 1.2)

        return e2e_p95

    def _generate_reasoning(
        self,
        model: ModelInfo,
        bench: BenchmarkData,
        gpu_config: GPUConfig,
        intent: DeploymentIntent
    ) -> str:
        """Generate explanation for recommendation."""
        reasons = []

        # Model selection
        reasons.append(f"Selected {model.name} ({model.size_parameters}) for {intent.use_case} use case")

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
        reasons.append(
            f"Expected performance: TTFT={bench.ttft_p90_ms}ms, TPOT={bench.tpot_p90_ms}ms"
        )

        return ". ".join(reasons)

    def _select_best_config(
        self,
        configs: List[Tuple[DeploymentRecommendation, float]],
        budget_constraint: str
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
                configs,
                key=lambda x: (x[0].predicted_ttft_p90_ms + x[0].predicted_tpot_p90_ms)
            )
            return configs_by_perf[0][0]
