"""Data access layer for model benchmark data using PostgreSQL."""

import logging
import os
from typing import Optional

import psycopg2
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)


class BenchmarkData:
    """Model performance benchmark entry."""

    def __init__(self, data: dict):
        """Initialize from database row dict."""
        self.model_hf_repo = data["model_hf_repo"]
        self.hardware = data["hardware"]
        self.hardware_count = data["hardware_count"]
        self.framework = data.get("framework", "vllm")
        self.framework_version = data.get("framework_version", "0.6.2")

        # Traffic profile
        self.prompt_tokens = data["prompt_tokens"]
        self.output_tokens = data["output_tokens"]
        self.mean_input_tokens = data["mean_input_tokens"]
        self.mean_output_tokens = data["mean_output_tokens"]

        # TTFT metrics (p95)
        self.ttft_mean = data["ttft_mean"]
        self.ttft_p90 = data["ttft_p90"]
        self.ttft_p95 = data["ttft_p95"]
        self.ttft_p99 = data["ttft_p99"]

        # ITL metrics (p95)
        self.itl_mean = data.get("itl_mean")
        self.itl_p90 = data.get("itl_p90")
        self.itl_p95 = data.get("itl_p95")
        self.itl_p99 = data.get("itl_p99")

        # E2E metrics (p95)
        self.e2e_mean = data["e2e_mean"]
        self.e2e_p90 = data["e2e_p90"]
        self.e2e_p95 = data["e2e_p95"]
        self.e2e_p99 = data["e2e_p99"]

        # Throughput
        self.tokens_per_second = data["tokens_per_second"]
        self.requests_per_second = data["requests_per_second"]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "model_hf_repo": self.model_hf_repo,
            "hardware": self.hardware,
            "hardware_count": self.hardware_count,
            "framework": self.framework,
            "framework_version": self.framework_version,
            "prompt_tokens": self.prompt_tokens,
            "output_tokens": self.output_tokens,
            "mean_input_tokens": self.mean_input_tokens,
            "mean_output_tokens": self.mean_output_tokens,
            "ttft_mean": self.ttft_mean,
            "ttft_p90": self.ttft_p90,
            "ttft_p95": self.ttft_p95,
            "ttft_p99": self.ttft_p99,
            "itl_mean": self.itl_mean,
            "itl_p90": self.itl_p90,
            "itl_p95": self.itl_p95,
            "itl_p99": self.itl_p99,
            "e2e_mean": self.e2e_mean,
            "e2e_p90": self.e2e_p90,
            "e2e_p95": self.e2e_p95,
            "e2e_p99": self.e2e_p99,
            "tokens_per_second": self.tokens_per_second,
            "requests_per_second": self.requests_per_second,
        }


class BenchmarkRepository:
    """Repository for querying model benchmark data from PostgreSQL."""

    def __init__(self, database_url: Optional[str] = None):
        """
        Initialize benchmark repository.

        Args:
            database_url: PostgreSQL connection string (defaults to DATABASE_URL env var)
        """
        self.database_url = database_url or os.getenv(
            "DATABASE_URL",
            "postgresql://postgres:compass@localhost:5432/compass"
        )
        self._test_connection()

    def _test_connection(self):
        """Test database connection on initialization."""
        try:
            conn = self._get_connection()
            conn.close()
            logger.info("Successfully connected to PostgreSQL benchmark database")
        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            logger.error(f"Database URL: {self.database_url}")
            raise

    def _get_connection(self):
        """Get a database connection."""
        return psycopg2.connect(self.database_url, cursor_factory=RealDictCursor)

    def get_benchmark(
        self,
        model_hf_repo: str,
        hardware: str,
        hardware_count: int,
        prompt_tokens: int,
        output_tokens: int
    ) -> Optional[BenchmarkData]:
        """
        Get benchmark for specific configuration and traffic profile.

        Args:
            model_hf_repo: Model HuggingFace repository
            hardware: GPU type (e.g., NVIDIA-L4, NVIDIA-A100-80GB)
            hardware_count: Number of GPUs (tensor parallel size)
            prompt_tokens: Target prompt length (GuideLLM config)
            output_tokens: Target output length (GuideLLM config)

        Returns:
            BenchmarkData if found, None otherwise
        """
        query = """
            SELECT * FROM exported_summaries
            WHERE model_hf_repo = %s
              AND hardware = %s
              AND hardware_count = %s
              AND prompt_tokens = %s
              AND output_tokens = %s
            LIMIT 1
        """

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, (model_hf_repo, hardware, hardware_count, prompt_tokens, output_tokens))
            row = cursor.fetchone()
            cursor.close()

            if row:
                return BenchmarkData(dict(row))
            return None
        finally:
            conn.close()

    def get_benchmarks_for_traffic_profile(
        self,
        model_hf_repo: str,
        hardware: str,
        hardware_count: int,
        prompt_tokens: int,
        output_tokens: int
    ) -> list[BenchmarkData]:
        """
        Get all benchmarks matching model, hardware, and traffic profile.

        This is similar to get_benchmark but returns a list in case there are
        multiple benchmark runs for the same configuration.

        Args:
            model_hf_repo: Model HuggingFace repository
            hardware: GPU type
            hardware_count: Number of GPUs
            prompt_tokens: Target prompt length
            output_tokens: Target output length

        Returns:
            List of matching benchmarks
        """
        query = """
            SELECT * FROM exported_summaries
            WHERE model_hf_repo = %s
              AND hardware = %s
              AND hardware_count = %s
              AND prompt_tokens = %s
              AND output_tokens = %s
        """

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, (model_hf_repo, hardware, hardware_count, prompt_tokens, output_tokens))
            rows = cursor.fetchall()
            cursor.close()

            return [BenchmarkData(dict(row)) for row in rows]
        finally:
            conn.close()

    def get_benchmarks_for_model(self, model_hf_repo: str) -> list[BenchmarkData]:
        """
        Get all benchmarks for a specific model.

        Args:
            model_hf_repo: Model HuggingFace repository

        Returns:
            List of benchmarks for this model
        """
        query = """
            SELECT * FROM exported_summaries
            WHERE model_hf_repo = %s
            ORDER BY hardware, hardware_count, prompt_tokens, output_tokens
        """

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, (model_hf_repo,))
            rows = cursor.fetchall()
            cursor.close()

            return [BenchmarkData(dict(row)) for row in rows]
        finally:
            conn.close()

    def get_benchmarks_for_hardware(self, hardware: str) -> list[BenchmarkData]:
        """
        Get all benchmarks for a specific GPU type.

        Args:
            hardware: GPU type (e.g., NVIDIA-A100-80GB)

        Returns:
            List of benchmarks for this hardware
        """
        query = """
            SELECT * FROM exported_summaries
            WHERE hardware = %s
            ORDER BY model_hf_repo, hardware_count, prompt_tokens, output_tokens
        """

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query, (hardware,))
            rows = cursor.fetchall()
            cursor.close()

            return [BenchmarkData(dict(row)) for row in rows]
        finally:
            conn.close()

    def find_configurations_meeting_slo(
        self,
        prompt_tokens: int,
        output_tokens: int,
        ttft_p95_max_ms: int,
        itl_p95_max_ms: int,
        e2e_p95_max_ms: int,
        min_qps: float = 0
    ) -> list[BenchmarkData]:
        """
        Find all configurations that meet SLO requirements for a traffic profile.

        For each unique system configuration (model_hf_repo, hardware, hardware_count,
        prompt_tokens, output_tokens), selects only the benchmark with the highest
        requests_per_second that still meets SLO requirements. This is critical because
        benchmarks are collected at multiple QPS rates, and we want the maximum throughput
        that doesn't violate SLO targets.

        Args:
            prompt_tokens: Target prompt length
            output_tokens: Target output length
            ttft_p95_max_ms: Maximum acceptable TTFT p95 (ms)
            itl_p95_max_ms: Maximum acceptable ITL p95 (ms/token)
            e2e_p95_max_ms: Maximum acceptable E2E p95 (ms)
            min_qps: Minimum required QPS

        Returns:
            List of benchmarks meeting all criteria (one per system configuration)
        """
        # Use window function to rank benchmarks by requests_per_second within each
        # system configuration, then select only the highest QPS that meets SLO.
        # When multiple benchmarks exist at the same QPS, prefer the one with lowest E2E latency.
        query = """
            WITH ranked_configs AS (
                SELECT *,
                       ROW_NUMBER() OVER (
                           PARTITION BY model_hf_repo, hardware, hardware_count
                           ORDER BY requests_per_second DESC, e2e_p95 ASC
                       ) as rn
                FROM exported_summaries
                WHERE prompt_tokens = %s
                  AND output_tokens = %s
                  AND ttft_p95 <= %s
                  AND itl_p95 <= %s
                  AND e2e_p95 <= %s
                  AND requests_per_second >= %s
            )
            SELECT
                id, config_id, model_hf_repo, provider, type,
                ttft_mean, ttft_p90, ttft_p95, ttft_p99,
                e2e_mean, e2e_p90, e2e_p95, e2e_p99,
                itl_mean, itl_p90, itl_p95, itl_p99,
                tps_mean, tps_p90, tps_p95, tps_p99,
                hardware, hardware_count, framework, requests_per_second, tokens_per_second,
                mean_input_tokens, mean_output_tokens,
                prompt_tokens, output_tokens
            FROM ranked_configs
            WHERE rn = 1
            ORDER BY model_hf_repo, hardware, hardware_count
        """

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                query,
                (prompt_tokens, output_tokens, ttft_p95_max_ms, itl_p95_max_ms, e2e_p95_max_ms, min_qps)
            )
            rows = cursor.fetchall()
            cursor.close()

            return [BenchmarkData(dict(row)) for row in rows]
        finally:
            conn.close()

    def get_available_models(self) -> list[str]:
        """Get list of all available models in the database."""
        query = "SELECT DISTINCT model_hf_repo FROM exported_summaries ORDER BY model_hf_repo"

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()

            return [row["model_hf_repo"] for row in rows]
        finally:
            conn.close()

    def get_available_hardware_types(self) -> list[str]:
        """Get list of all available hardware types in the database."""
        query = "SELECT DISTINCT hardware FROM exported_summaries ORDER BY hardware"

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()

            return [row["hardware"] for row in rows]
        finally:
            conn.close()

    def get_traffic_profiles(self) -> list[tuple[int, int]]:
        """Get list of all available traffic profiles (prompt_tokens, output_tokens)."""
        query = """
            SELECT DISTINCT prompt_tokens, output_tokens
            FROM exported_summaries
            ORDER BY prompt_tokens, output_tokens
        """

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()

            return [(row["prompt_tokens"], row["output_tokens"]) for row in rows]
        finally:
            conn.close()

    def get_all_benchmarks(self) -> list[BenchmarkData]:
        """
        Get all benchmarks.

        Warning: This may return a large dataset. Use with caution.
        """
        query = """
            SELECT * FROM exported_summaries
            ORDER BY model_hf_repo, hardware, hardware_count, prompt_tokens, output_tokens
        """

        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(query)
            rows = cursor.fetchall()
            cursor.close()

            return [BenchmarkData(dict(row)) for row in rows]
        finally:
            conn.close()
