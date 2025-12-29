"""Solution scoring for multi-criteria recommendation ranking.

Scores deployment configurations on 4 criteria (0-100 scale):
- Accuracy/Quality: Model capability (from Artificial Analysis benchmarks or param count fallback)
- Price: Cost efficiency (inverse of cost, normalized)  
- Latency: SLO compliance and headroom (from Andre's PostgreSQL benchmarks)
- Complexity: Deployment simplicity (fewer GPUs = simpler)

INTEGRATION NOTE:
- Quality scoring: Uses Yuval's weighted_scores CSVs (Artificial Analysis benchmarks)
- Latency/Price/Complexity: Uses Andre's scoring logic and benchmark data
"""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Try to import use-case quality scorer
try:
    from .usecase_quality_scorer import score_model_quality
    USE_CASE_QUALITY_AVAILABLE = True
except ImportError:
    USE_CASE_QUALITY_AVAILABLE = False


class SolutionScorer:
    """Score deployment configurations on 4 criteria (0-100 scale)."""

    # Accuracy tiers based on model parameter count (in billions)
    # Larger models generally have higher accuracy/capability
    ACCURACY_TIERS = {
        3: 40,
        4: 45,
        7: 55,
        8: 60,
        9: 62,
        14: 70,
        17: 72,
        20: 75,
        24: 78,
        27: 80,
        70: 85,
        120: 90,
        405: 95,
        480: 98,
    }

    # Complexity scores based on total GPU count
    COMPLEXITY_SCORES = {
        1: 100,
        2: 90,
        3: 82,
        4: 75,
        5: 70,
        6: 65,
        7: 62,
        8: 60,
    }

    # Default weights for balanced score
    DEFAULT_WEIGHTS = {
        "accuracy": 0.40,
        "price": 0.40,
        "latency": 0.10,
        "complexity": 0.10,
    }

    def score_accuracy(self, model_size_str: str, model_name: Optional[str] = None, 
                        use_case: Optional[str] = None) -> int:
        """
        Score model accuracy/quality.
        
        Priority:
        1. Use-case specific benchmark score (Artificial Analysis data) if available
        2. Fallback to model size-based heuristic (Andre's original logic)

        Args:
            model_size_str: Model size string (e.g., "8B", "70B", "8x7B")
            model_name: Optional model name for use-case-specific scoring
            use_case: Optional use case for benchmark-based scoring

        Returns:
            Score 0-100
        """
        # Try use-case-specific quality scoring first (Yuval's contribution)
        if USE_CASE_QUALITY_AVAILABLE and model_name and use_case:
            quality_score = score_model_quality(model_name, use_case)
            if quality_score > 0:
                logger.debug(f"Quality score for {model_name} ({use_case}): {quality_score:.1f}")
                return int(quality_score)
        
        # Fallback to size-based heuristic (Andre's original logic)
        return self._score_accuracy_by_size(model_size_str)
    
    def _score_accuracy_by_size(self, model_size_str: str) -> int:
        """
        Score model accuracy based on parameter count tier (fallback).

        Args:
            model_size_str: Model size string (e.g., "8B", "70B", "8x7B")

        Returns:
            Score 0-100
        """
        param_count = self._extract_param_count(model_size_str)

        # Find the closest tier at or below the param count
        best_score = 40  # minimum score
        for tier_size, tier_score in sorted(self.ACCURACY_TIERS.items()):
            if param_count >= tier_size:
                best_score = tier_score
            else:
                break

        logger.debug(f"Accuracy score for {model_size_str} ({param_count}B): {best_score}")
        return best_score

    def score_price(self, cost_per_month: float, min_cost: float, max_cost: float) -> int:
        """
        Score price using non-linear formula for better differentiation.

        Enhanced Formula: 100 * (1 - (Monthly_Cost / Max_Monthly_Cost)^0.7)
        
        This creates more spread between configurations:
        - 1x A100: ~$1,100/mo → Score: 95
        - 2x A100: ~$2,200/mo → Score: 85  
        - 4x H100: ~$7,900/mo → Score: 60
        - 8x H100: ~$15,800/mo → Score: 35
        
        The power of 0.7 creates non-linear scaling that:
        - Rewards cheaper configurations more significantly
        - Creates meaningful gaps between similar-cost options
        - Penalizes expensive multi-GPU setups appropriately

        Args:
            cost_per_month: Configuration cost in USD/month
            min_cost: Minimum cost among all configurations
            max_cost: Maximum cost among all configurations

        Returns:
            Score 0-100 (100 = cheapest, 0 = most expensive)
        """
        import math
        
        if max_cost == 0:
            return 100
            
        if max_cost == min_cost:
            # All configs have same cost - give them high score
            return 95

        # Clamp cost to range
        cost = max(min_cost, min(max_cost, cost_per_month))
        
        # Non-linear scoring formula
        # Power of 0.7 creates better spread than linear
        cost_ratio = cost / max_cost
        score = int(100 * (1 - math.pow(cost_ratio, 0.7)))
        
        # Ensure minimum score of 5 for any valid config
        score = max(5, min(100, score))
        
        logger.debug(
            f"Price score for ${cost_per_month:,.0f}/mo: {score} "
            f"(ratio: {cost_ratio:.2f}, min: ${min_cost:,.0f}, max: ${max_cost:,.0f})"
        )
        return score

    # Dynamic benchmark thresholds per use case (from slo_ranges.json real_data)
    # Format: {use_case: {metric: {excellent: p50, good: p90, acceptable: max/2, max: max}}}
    LATENCY_BENCHMARKS_BY_USE_CASE = {
        # 512/256 token configs - fast interactive use cases
        'chatbot_conversational': {
            'ttft': {'excellent': 127, 'good': 384, 'acceptable': 3000, 'max': 11720},
            'itl': {'excellent': 21, 'good': 122, 'acceptable': 150, 'max': 171},
            'e2e': {'excellent': 6338, 'good': 39886, 'acceptable': 50000, 'max': 54632},
            'tps': {'excellent': 5000, 'good': 2000, 'acceptable': 500, 'min': 50},
        },
        'code_completion': {
            'ttft': {'excellent': 127, 'good': 384, 'acceptable': 3000, 'max': 11720},
            'itl': {'excellent': 21, 'good': 122, 'acceptable': 150, 'max': 171},
            'e2e': {'excellent': 6338, 'good': 39886, 'acceptable': 50000, 'max': 54632},
            'tps': {'excellent': 5000, 'good': 2000, 'acceptable': 500, 'min': 50},
        },
        'translation': {
            'ttft': {'excellent': 127, 'good': 384, 'acceptable': 3000, 'max': 11720},
            'itl': {'excellent': 21, 'good': 122, 'acceptable': 150, 'max': 171},
            'e2e': {'excellent': 6338, 'good': 39886, 'acceptable': 50000, 'max': 54632},
            'tps': {'excellent': 5000, 'good': 2000, 'acceptable': 500, 'min': 50},
        },
        'content_creation': {
            'ttft': {'excellent': 127, 'good': 384, 'acceptable': 3000, 'max': 11720},
            'itl': {'excellent': 21, 'good': 122, 'acceptable': 150, 'max': 171},
            'e2e': {'excellent': 6338, 'good': 39886, 'acceptable': 50000, 'max': 54632},
            'tps': {'excellent': 5000, 'good': 2000, 'acceptable': 500, 'min': 50},
        },
        # 1024/1024 token config - medium
        'code_generation_detailed': {
            'ttft': {'excellent': 150, 'good': 500, 'acceptable': 700, 'max': 742},
            'itl': {'excellent': 10, 'good': 40, 'acceptable': 55, 'max': 63},
            'e2e': {'excellent': 12000, 'good': 45000, 'acceptable': 60000, 'max': 67323},
            'tps': {'excellent': 3000, 'good': 1500, 'acceptable': 400, 'min': 50},
        },
        # 4096/512 token config - longer context
        'summarization_short': {
            'ttft': {'excellent': 200, 'good': 1000, 'acceptable': 50000, 'max': 200000},
            'itl': {'excellent': 30, 'good': 100, 'acceptable': 150, 'max': 200},
            'e2e': {'excellent': 15000, 'good': 100000, 'acceptable': 200000, 'max': 281786},
            'tps': {'excellent': 2000, 'good': 1000, 'acceptable': 300, 'min': 30},
        },
        'document_analysis_rag': {
            'ttft': {'excellent': 200, 'good': 1000, 'acceptable': 50000, 'max': 200000},
            'itl': {'excellent': 30, 'good': 100, 'acceptable': 150, 'max': 200},
            'e2e': {'excellent': 15000, 'good': 100000, 'acceptable': 200000, 'max': 281786},
            'tps': {'excellent': 2000, 'good': 1000, 'acceptable': 300, 'min': 30},
        },
        # 10240/1536 token config - very long documents
        'long_document_summarization': {
            'ttft': {'excellent': 400, 'good': 800, 'acceptable': 1200, 'max': 1305},
            'itl': {'excellent': 15, 'good': 35, 'acceptable': 45, 'max': 50},
            'e2e': {'excellent': 25000, 'good': 100000, 'acceptable': 150000, 'max': 185299},
            'tps': {'excellent': 1000, 'good': 500, 'acceptable': 200, 'min': 20},
        },
        'research_legal_analysis': {
            'ttft': {'excellent': 400, 'good': 800, 'acceptable': 1200, 'max': 1305},
            'itl': {'excellent': 15, 'good': 35, 'acceptable': 45, 'max': 50},
            'e2e': {'excellent': 25000, 'good': 100000, 'acceptable': 150000, 'max': 185299},
            'tps': {'excellent': 1000, 'good': 500, 'acceptable': 200, 'min': 20},
        },
    }
    
    # Default fallback for unknown use cases
    LATENCY_BENCHMARKS_DEFAULT = {
        'ttft': {'excellent': 100, 'good': 300, 'acceptable': 1000, 'max': 15000},
        'itl': {'excellent': 15, 'good': 30, 'acceptable': 60, 'max': 200},
        'e2e': {'excellent': 2000, 'good': 5000, 'acceptable': 15000, 'max': 60000},
        'tps': {'excellent': 5000, 'good': 2000, 'acceptable': 500, 'min': 50},
    }

    def score_latency(
        self,
        predicted_ttft_ms: int,
        predicted_itl_ms: int,
        predicted_e2e_ms: int,
        target_ttft_ms: int,
        target_itl_ms: int,
        target_e2e_ms: int,
        throughput_tps: float = 0,
        use_case: str = None,
    ) -> tuple[int, str]:
        """
        Score latency using ABSOLUTE PERFORMANCE + SLO HEADROOM.
        
        Two-factor scoring approach (enterprise standard):
        1. Absolute Performance (60%): How fast is the model compared to benchmarks?
        2. SLO Headroom (40%): How much margin does it have vs the target?
        
        Benchmarks are DYNAMIC per use case - comparing a chatbot (512/256 tokens)
        to a long doc summarization (10240/1536 tokens) uses different thresholds.

        Args:
            predicted_ttft_ms: Predicted TTFT p95 in ms
            predicted_itl_ms: Predicted ITL p95 in ms
            predicted_e2e_ms: Predicted E2E p95 in ms
            target_ttft_ms: Target TTFT p95 in ms
            target_itl_ms: Target ITL p95 in ms
            target_e2e_ms: Target E2E p95 in ms
            throughput_tps: Optional throughput in tokens/second
            use_case: Optional use case for dynamic benchmark selection

        Returns:
            Tuple of (score 0-100, slo_status)
            - slo_status: "compliant", "near_miss", or "exceeds"
        """
        import math
        
        # Get use-case specific benchmarks (or default)
        benchmarks = self.LATENCY_BENCHMARKS_BY_USE_CASE.get(
            use_case, self.LATENCY_BENCHMARKS_DEFAULT
        )
        
        # ===== STEP 1: Calculate SLO compliance ratios =====
        ratios = []
        if target_ttft_ms > 0:
            ratios.append(predicted_ttft_ms / target_ttft_ms)
        if target_itl_ms > 0:
            ratios.append(predicted_itl_ms / target_itl_ms)
        if target_e2e_ms > 0:
            ratios.append(predicted_e2e_ms / target_e2e_ms)

        if not ratios:
            return 75, "compliant"

        worst_ratio = max(ratios)
        
        # Determine SLO status
        if worst_ratio <= 1.0:
            slo_status = "compliant"
        elif worst_ratio <= 1.2:
            slo_status = "near_miss"
        else:
            slo_status = "exceeds"
            # Return low score for non-compliant
            score = max(0, int(30 - (worst_ratio - 1.0) * 20))
            return score, slo_status

        # ===== STEP 2: Calculate ABSOLUTE performance score (60% weight) =====
        # Uses USE-CASE SPECIFIC benchmark reference values
        
        def score_metric_lower_better(value: float, metric_benchmarks: dict) -> float:
            """Score a metric where LOWER is better (TTFT, ITL, E2E)."""
            if value <= metric_benchmarks['excellent']:
                # Excellent: 85-100
                return 100 - (value / metric_benchmarks['excellent']) * 15
            elif value <= metric_benchmarks['good']:
                # Good: 70-85
                progress = (value - metric_benchmarks['excellent']) / (metric_benchmarks['good'] - metric_benchmarks['excellent'])
                return 85 - progress * 15
            elif value <= metric_benchmarks['acceptable']:
                # Acceptable: 50-70
                progress = (value - metric_benchmarks['good']) / (metric_benchmarks['acceptable'] - metric_benchmarks['good'])
                return 70 - progress * 20
            else:
                # Below acceptable: 20-50
                progress = min(1.0, (value - metric_benchmarks['acceptable']) / (metric_benchmarks['max'] - metric_benchmarks['acceptable']))
                return 50 - progress * 30

        def score_metric_higher_better(value: float, metric_benchmarks: dict) -> float:
            """Score a metric where HIGHER is better (TPS/throughput)."""
            if value >= metric_benchmarks['excellent']:
                # Excellent: 85-100
                return 85 + min(15, (value - metric_benchmarks['excellent']) / metric_benchmarks['excellent'] * 15)
            elif value >= metric_benchmarks['good']:
                # Good: 70-85
                progress = (value - metric_benchmarks['good']) / (metric_benchmarks['excellent'] - metric_benchmarks['good'])
                return 70 + progress * 15
            elif value >= metric_benchmarks['acceptable']:
                # Acceptable: 50-70
                progress = (value - metric_benchmarks['acceptable']) / (metric_benchmarks['good'] - metric_benchmarks['acceptable'])
                return 50 + progress * 20
            else:
                # Below acceptable: 20-50
                progress = max(0, (value - metric_benchmarks['min']) / (metric_benchmarks['acceptable'] - metric_benchmarks['min']))
                return 20 + progress * 30

        ttft_score = score_metric_lower_better(predicted_ttft_ms, benchmarks['ttft'])
        itl_score = score_metric_lower_better(predicted_itl_ms, benchmarks['itl'])
        e2e_score = score_metric_lower_better(predicted_e2e_ms, benchmarks['e2e'])
        
        # Add throughput scoring if provided
        if throughput_tps > 0:
            tps_score = score_metric_higher_better(throughput_tps, benchmarks['tps'])
            # Weight: TTFT 25%, ITL 25%, E2E 25%, TPS 25%
            absolute_score = ttft_score * 0.25 + itl_score * 0.25 + e2e_score * 0.25 + tps_score * 0.25
        else:
            # No throughput - use original weights
            # Weight: TTFT 35%, ITL 30%, E2E 35%
            absolute_score = ttft_score * 0.35 + itl_score * 0.30 + e2e_score * 0.35
            tps_score = 0
        
        # ===== STEP 3: Calculate HEADROOM score (40% weight) =====
        # How much margin vs SLO target?
        avg_ratio = sum(ratios) / len(ratios)
        
        # Convert ratio to score: ratio 0.1 → 100, ratio 1.0 → 50
        headroom_score = 100 - (avg_ratio * 50)
        headroom_score = max(50, min(100, headroom_score))
        
        # ===== STEP 4: Combine scores =====
        # 60% absolute performance + 40% SLO headroom
        final_score = absolute_score * 0.60 + headroom_score * 0.40
        
        # Near-miss penalty
        if slo_status == "near_miss":
            final_score = min(49, final_score * 0.7)
        
        score = int(max(20, min(100, final_score)))

        logger.debug(
            f"Latency score: {score} ({slo_status}) [use_case={use_case or 'default'}] - "
            f"Absolute: {absolute_score:.0f} (TTFT={ttft_score:.0f}, ITL={itl_score:.0f}, E2E={e2e_score:.0f}, TPS={tps_score:.0f}), "
            f"Headroom: {headroom_score:.0f}, "
            f"Predicted: TTFT={predicted_ttft_ms}, ITL={predicted_itl_ms}, E2E={predicted_e2e_ms}, TPS={throughput_tps:.0f}"
        )
        return score, slo_status

    def score_complexity(self, total_gpu_count: int) -> int:
        """
        Score complexity based on deployment topology.

        Args:
            total_gpu_count: Total GPUs required (tensor_parallel * replicas)

        Returns:
            Score 0-100 (100 = simplest, lower = more complex)
        """
        # Use predefined scores or calculate for larger counts
        if total_gpu_count in self.COMPLEXITY_SCORES:
            score = self.COMPLEXITY_SCORES[total_gpu_count]
        elif total_gpu_count > 8:
            # Linear decay for very large deployments
            score = max(40, 60 - (total_gpu_count - 8) * 2)
        else:
            score = 60

        logger.debug(f"Complexity score for {total_gpu_count} GPUs: {score}")
        return score

    def score_balanced(
        self,
        accuracy_score: int,
        price_score: int,
        latency_score: int,
        complexity_score: int,
        weights: Optional[dict] = None,
    ) -> float:
        """
        Calculate weighted composite score.

        Args:
            accuracy_score: Accuracy score (0-100)
            price_score: Price score (0-100)
            latency_score: Latency score (0-100)
            complexity_score: Complexity score (0-100)
            weights: Optional custom weights (default: 40% accuracy, 40% price,
                     10% latency, 10% complexity)

        Returns:
            Weighted composite score (0-100)
        """
        w = weights or self.DEFAULT_WEIGHTS

        balanced = (
            accuracy_score * w["accuracy"]
            + price_score * w["price"]
            + latency_score * w["latency"]
            + complexity_score * w["complexity"]
        )

        logger.debug(
            f"Balanced score: {balanced:.1f} "
            f"(A={accuracy_score}, P={price_score}, L={latency_score}, C={complexity_score})"
        )
        return round(balanced, 1)

    def _extract_param_count(self, size_str: str) -> float:
        """
        Extract approximate parameter count from size string.

        Args:
            size_str: Size string (e.g., "8B", "70B", "8x7B")

        Returns:
            Approximate parameter count in billions
        """
        try:
            # Handle "8B", "70B" format
            if "B" in size_str and "x" not in size_str.lower():
                # Extract number before B
                match = re.search(r"(\d+\.?\d*)\s*B", size_str, re.IGNORECASE)
                if match:
                    return float(match.group(1))

            # Handle "8x7B" MoE format (use total params)
            if "x" in size_str.lower() and "B" in size_str.upper():
                match = re.search(r"(\d+)\s*x\s*(\d+\.?\d*)\s*B", size_str, re.IGNORECASE)
                if match:
                    return float(match.group(1)) * float(match.group(2))

            # Fallback: try to extract any number
            match = re.search(r"(\d+\.?\d*)", size_str)
            if match:
                return float(match.group(1))

            return 10.0  # Default fallback
        except Exception:
            logger.warning(f"Could not parse size string: {size_str}")
            return 10.0
