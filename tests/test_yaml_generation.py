"""Test script for Sprint 4: YAML generation and deployment functionality.

This script tests the complete workflow:
1. Generate a recommendation
2. Generate YAML deployment files
3. Validate the generated YAMLs
4. Fetch mock deployment status
"""

import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.context_intent.schema import (
    DeploymentIntent,
    TrafficProfile,
    SLOTargets,
    GPUConfig,
    DeploymentRecommendation
)
from src.deployment.generator import DeploymentGenerator
from src.deployment.validator import YAMLValidator, ValidationError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_test_recommendation() -> DeploymentRecommendation:
    """Create a test recommendation for a chatbot deployment."""

    intent = DeploymentIntent(
        use_case="chatbot",
        user_count=5000,
        latency_requirement="high",
        throughput_priority="high",
        budget_constraint="moderate",
        domain_specialization=["general"]
    )

    traffic_profile = TrafficProfile(
        prompt_tokens_mean=150,
        generation_tokens_mean=200,
        expected_qps=50.0
    )

    slo_targets = SLOTargets(
        ttft_p90_target_ms=200,
        tpot_p90_target_ms=50,
        e2e_p90_target_ms=2000
    )

    gpu_config = GPUConfig(
        gpu_type="NVIDIA-A100-80GB",
        gpu_count=2,
        tensor_parallel=2,
        replicas=1
    )

    recommendation = DeploymentRecommendation(
        intent=intent,
        traffic_profile=traffic_profile,
        slo_targets=slo_targets,
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        model_name="Llama 3.1 8B Instruct",
        gpu_config=gpu_config,
        predicted_ttft_p90_ms=185,
        predicted_tpot_p90_ms=48,
        predicted_e2e_p90_ms=1850,
        predicted_throughput_qps=122.0,
        cost_per_hour_usd=8.20,
        cost_per_month_usd=5986.0,
        meets_slo=True,
        reasoning="Llama 3.1 8B Instruct provides excellent latency for chatbot use cases. "
                  "2x A100-80GB GPUs in tensor parallel configuration meets all SLO targets "
                  "with headroom for traffic spikes. Cost-effective for 5000 concurrent users."
    )

    return recommendation


def test_yaml_generation():
    """Test YAML generation and validation."""

    logger.info("=" * 80)
    logger.info("SPRINT 4 TEST: YAML Generation and Deployment")
    logger.info("=" * 80)

    # Step 1: Create test recommendation
    logger.info("\n[1/4] Creating test recommendation...")
    recommendation = create_test_recommendation()
    logger.info(f"✓ Recommendation created: {recommendation.model_name} on "
                f"{recommendation.gpu_config.gpu_count}x {recommendation.gpu_config.gpu_type}")

    # Step 2: Generate YAML files
    logger.info("\n[2/4] Generating deployment YAML files...")
    generator = DeploymentGenerator()

    try:
        result = generator.generate_all(recommendation, namespace="default")
        logger.info(f"✓ Deployment ID: {result['deployment_id']}")
        logger.info(f"✓ Namespace: {result['namespace']}")
        logger.info("✓ Generated files:")
        for config_type, file_path in result['files'].items():
            logger.info(f"  - {config_type}: {file_path}")
    except Exception as e:
        logger.error(f"✗ YAML generation failed: {e}")
        return False

    # Step 3: Validate generated YAMLs
    logger.info("\n[3/4] Validating generated YAML files...")
    validator = YAMLValidator()

    try:
        validation_results = validator.validate_all(result['files'])
        logger.info("✓ All YAML files validated successfully:")
        for config_type, valid in validation_results.items():
            logger.info(f"  - {config_type}: {'✓ VALID' if valid else '✗ INVALID'}")
    except ValidationError as e:
        logger.error(f"✗ Validation failed: {e}")
        return False

    # Step 4: Display sample YAML content
    logger.info("\n[4/4] Sample YAML content (KServe InferenceService):")
    logger.info("-" * 80)

    inferenceservice_path = result['files'].get('inferenceservice')
    if inferenceservice_path:
        try:
            with open(inferenceservice_path, 'r') as f:
                content = f.read()
                # Show first 40 lines
                lines = content.split('\n')[:40]
                for line in lines:
                    logger.info(line)
                if len(content.split('\n')) > 40:
                    logger.info("... (truncated)")
        except Exception as e:
            logger.warning(f"Could not read file: {e}")

    logger.info("-" * 80)

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SPRINT 4 TEST RESULTS")
    logger.info("=" * 80)
    logger.info("✓ Recommendation generation: PASSED")
    logger.info("✓ YAML generation: PASSED")
    logger.info("✓ YAML validation: PASSED")
    logger.info("✓ Files written to: generated_configs/")
    logger.info("=" * 80)
    logger.info("\nSprint 4 implementation complete! 🎉")
    logger.info("\nNext steps:")
    logger.info("1. Start the FastAPI backend: ./run_api.sh")
    logger.info("2. Start the Streamlit UI: ./run_ui.sh")
    logger.info("3. Test the full workflow in the UI:")
    logger.info("   - Get a recommendation")
    logger.info("   - Click 'Generate Deployment YAML' on the Cost tab")
    logger.info("   - View the Monitoring tab to see simulated observability")
    logger.info("=" * 80)

    return True


if __name__ == "__main__":
    success = test_yaml_generation()
    sys.exit(0 if success else 1)
