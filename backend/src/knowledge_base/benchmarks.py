"""Data access layer for model benchmark data."""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict

logger = logging.getLogger(__name__)


class BenchmarkData:
    """Model performance benchmark entry."""

    def __init__(self, data: dict):
        self.model_id = data["model_id"]
        self.gpu_type = data["gpu_type"]
        self.tensor_parallel = data["tensor_parallel"]
        self.ttft_p50_ms = data["ttft_p50_ms"]
        self.ttft_p90_ms = data["ttft_p90_ms"]
        self.ttft_p99_ms = data["ttft_p99_ms"]
        self.tpot_p50_ms = data["tpot_p50_ms"]
        self.tpot_p90_ms = data["tpot_p90_ms"]
        self.tpot_p99_ms = data["tpot_p99_ms"]
        self.throughput_tokens_per_sec = data["throughput_tokens_per_sec"]
        self.max_qps = data["max_qps"]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_id": self.model_id,
            "gpu_type": self.gpu_type,
            "tensor_parallel": self.tensor_parallel,
            "ttft_p50_ms": self.ttft_p50_ms,
            "ttft_p90_ms": self.ttft_p90_ms,
            "ttft_p99_ms": self.ttft_p99_ms,
            "tpot_p50_ms": self.tpot_p50_ms,
            "tpot_p90_ms": self.tpot_p90_ms,
            "tpot_p99_ms": self.tpot_p99_ms,
            "throughput_tokens_per_sec": self.throughput_tokens_per_sec,
            "max_qps": self.max_qps
        }


class BenchmarkRepository:
    """Repository for querying model benchmark data."""

    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize benchmark repository.

        Args:
            data_path: Path to benchmarks.json (defaults to ../data/benchmarks.json)
        """
        if data_path is None:
            # Default to data directory relative to this file
            data_path = Path(__file__).parent.parent.parent.parent / "data" / "benchmarks.json"

        self.data_path = data_path
        self._benchmarks: List[BenchmarkData] = []
        self._load_data()

    def _load_data(self):
        """Load benchmark data from JSON file."""
        try:
            with open(self.data_path) as f:
                data = json.load(f)
                self._benchmarks = [BenchmarkData(b) for b in data["benchmarks"]]
                logger.info(f"Loaded {len(self._benchmarks)} benchmark entries")
        except Exception as e:
            logger.error(f"Failed to load benchmarks from {self.data_path}: {e}")
            raise

    def get_benchmark(
        self,
        model_id: str,
        gpu_type: str,
        tensor_parallel: int = 1
    ) -> Optional[BenchmarkData]:
        """
        Get benchmark for specific model/GPU/TP configuration.

        Args:
            model_id: Model identifier
            gpu_type: GPU type (e.g., NVIDIA-L4)
            tensor_parallel: Tensor parallelism degree

        Returns:
            BenchmarkData if found, None otherwise
        """
        for bench in self._benchmarks:
            if (bench.model_id == model_id and
                bench.gpu_type == gpu_type and
                bench.tensor_parallel == tensor_parallel):
                return bench
        return None

    def get_benchmarks_for_model(self, model_id: str) -> List[BenchmarkData]:
        """
        Get all benchmarks for a specific model.

        Args:
            model_id: Model identifier

        Returns:
            List of benchmarks for this model
        """
        return [b for b in self._benchmarks if b.model_id == model_id]

    def get_benchmarks_for_gpu(self, gpu_type: str) -> List[BenchmarkData]:
        """
        Get all benchmarks for a specific GPU type.

        Args:
            gpu_type: GPU type

        Returns:
            List of benchmarks for this GPU
        """
        return [b for b in self._benchmarks if b.gpu_type == gpu_type]

    def find_configurations_meeting_slo(
        self,
        ttft_p90_max_ms: int,
        tpot_p90_max_ms: int,
        min_qps: float
    ) -> List[BenchmarkData]:
        """
        Find all configurations that meet SLO requirements.

        Args:
            ttft_p90_max_ms: Maximum acceptable TTFT p90 (ms)
            tpot_p90_max_ms: Maximum acceptable TPOT p90 (ms)
            min_qps: Minimum required QPS

        Returns:
            List of benchmarks meeting all criteria
        """
        results = []
        for bench in self._benchmarks:
            if (bench.ttft_p90_ms <= ttft_p90_max_ms and
                bench.tpot_p90_ms <= tpot_p90_max_ms and
                bench.max_qps >= min_qps):
                results.append(bench)
        return results

    def get_all_benchmarks(self) -> List[BenchmarkData]:
        """Get all benchmarks."""
        return self._benchmarks.copy()
