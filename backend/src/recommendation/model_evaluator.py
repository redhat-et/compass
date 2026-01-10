"""Model recommendation engine.

INTEGRATION NOTE (Yuval's Quality Scoring):
This module integrates use-case-specific quality scores from Artificial Analysis
benchmarks (weighted_scores CSVs). Quality scoring uses actual benchmark data
(MMLU-Pro, LiveCodeBench, IFBench, etc.) rather than model size heuristics.

Andre's latency/throughput benchmarks from PostgreSQL are kept as-is.
The final recommendation combines: Yuval's quality + Andre's latency/cost/complexity.
"""

import logging
from typing import Optional

from ..context_intent.schema import DeploymentIntent
from ..knowledge_base.model_catalog import ModelCatalog, ModelInfo

logger = logging.getLogger(__name__)

# Try to import use-case quality scorer (Yuval's contribution)
try:
    from .usecase_quality_scorer import score_model_quality, get_quality_scorer
    USE_CASE_QUALITY_AVAILABLE = True
    logger.info("Use-case quality scorer loaded (Artificial Analysis benchmarks)")
except ImportError:
    USE_CASE_QUALITY_AVAILABLE = False
    logger.warning("Use-case quality scorer not available, using size-based heuristics")


class ModelEvaluator:
    """Evaluate models for deployment intent and calculate accuracy scores.

    Quality Scoring:
    - Uses Artificial Analysis benchmark scores weighted by use case
      (e.g., code_completion uses LiveCodeBench 35%, SciCode 30%)
    - Fallback: Returns default score if quality data unavailable
    """

    def __init__(self, catalog: "Optional[ModelCatalog]" = None):
        """
        Initialize model evaluator.

        Args:
            catalog: Model catalog (creates default if not provided)
        """
        self.catalog = catalog or ModelCatalog()
        self._quality_scorer = get_quality_scorer() if USE_CASE_QUALITY_AVAILABLE else None

    def get_usecase_quality_score(self, model_name: str, use_case: str) -> float:
        """
        Get use-case-specific quality score from Artificial Analysis benchmarks.
        
        Args:
            model_name: Model name
            use_case: Use case identifier
            
        Returns:
            Quality score 0-100 (from weighted benchmarks or fallback heuristic)
        """
        if self._quality_scorer:
            return self._quality_scorer.get_quality_score(model_name, use_case)
        
        # Fallback to simple heuristic if quality scorer not available
        return 60.0  # Default moderate score

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
            if "B" in size_str and "x" not in size_str:
                return float(size_str.replace("B", ""))

            # Handle "8x7B" MoE format (approximate as total params)
            if "x" in size_str and "B" in size_str:
                parts = size_str.replace("B", "").split("x")
                return float(parts[0]) * float(parts[1])

            # Fallback
            return 10.0
        except Exception:
            logger.warning(f"Could not parse size string: {size_str}")
            return 10.0
