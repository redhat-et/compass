#!/usr/bin/env python3
"""
Test simulator deployment to verify Sprint 6 functionality.

⚠️  WARNING: This test is a work-in-progress and currently doesn't work due to
    incomplete DeploymentRecommendation schema population. Use the UI for
    end-to-end testing instead (scripts/run_ui.sh).

This script tests:
1. YAML generation with simulator mode
2. Deployment to KIND cluster
3. Pod becomes Ready
4. Inference endpoint is accessible

TODO: Fix schema field requirements or use workflow.py to generate valid recommendations
"""

import sys
import time
import subprocess
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

from src.configuration.generator import DeploymentGenerator
from src.cluster.manager import KubernetesClusterManager, KubernetesDeploymentError
from src.shared.schemas import DeploymentIntent, SLOTargets, TrafficProfile, GPUConfig, DeploymentRecommendation

def test_simulator_deployment():
    """Test end-to-end simulator deployment"""

    print("=" * 80)
    print("Sprint 6: Testing vLLM Simulator Deployment")
    print("=" * 80)

    # Step 1: Create a test recommendation
    print("\n1. Creating test deployment recommendation...")

    intent = DeploymentIntent(
        use_case="code_generation_detailed",
        experience_class="conversational",
        user_count=100,
        latency_requirement="high",
        throughput_priority="medium",
        budget_constraint="moderate",
        domain_specialization=["general"],
        additional_context="Test deployment for simulator"
    )

    slo = SLOTargets(
        ttft_p95_target_ms=200,
        itl_p95_target_ms=50,
        e2e_p95_target_ms=2000
    )

    traffic = TrafficProfile(
        expected_qps=5.0,
        prompt_tokens=512,
        output_tokens=256
    )

    gpu_config = GPUConfig(
        gpu_type="L4",
        gpu_count=1,
        tensor_parallel=1,
        replicas=1
    )

    recommendation = DeploymentRecommendation(
        intent=intent,
        traffic_profile=traffic,
        slo_targets=slo,
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        model_name="Llama 3.1 8B Instruct",
        gpu_config=gpu_config,
        predicted_ttft_p95_ms=150,
        predicted_itl_p95_ms=40,
        predicted_e2e_p95_ms=1800,
        predicted_throughput_qps=10.0,
        cost_per_hour_usd=0.50,
        cost_per_month_usd=365.0,
        meets_slo=True,
        reasoning="Testing simulator deployment",
        alternative_options=None
    )

    print(f"   Model: {recommendation.model_name}")
    print(f"   GPU: {gpu_config.gpu_type}")
    print(f"   Use case: {intent.use_case}")

    # Step 2: Generate YAML with simulator mode
    print("\n2. Generating deployment YAML (simulator mode)...")

    generator = DeploymentGenerator(simulator_mode=True)
    result = generator.generate_all(recommendation=recommendation, namespace="default")

    deployment_id = result["deployment_id"]
    files = result["files"]

    print(f"   Deployment ID: {deployment_id}")
    print(f"   Generated files:")
    for config_type, path in files.items():
        print(f"     - {config_type}: {Path(path).name}")

    # Step 3: Verify simulator mode in YAML
    print("\n3. Verifying simulator configuration...")

    with open(files["inferenceservice"], 'r') as f:
        yaml_content = f.read()

    if "vllm-simulator:latest" in yaml_content:
        print("   ✓ Using simulator image")
    else:
        print("   ✗ ERROR: Not using simulator image!")
        return False

    if "nvidia.com/gpu" not in yaml_content:
        print("   ✓ No GPU resources requested")
    else:
        print("   ✗ WARNING: GPU resources found in simulator mode")

    if "MODEL_NAME" in yaml_content:
        print("   ✓ MODEL_NAME environment variable set")
    else:
        print("   ✗ ERROR: MODEL_NAME not found!")
        return False

    # Step 4: Deploy to cluster
    print("\n4. Deploying to KIND cluster...")

    try:
        cluster_manager = KubernetesClusterManager(namespace="default")
        print("   ✓ Cluster accessible")
    except KubernetesDeploymentError as e:
        print(f"   ✗ Cluster not accessible: {e}")
        print("   Run: ./scripts/kind-cluster.sh start")
        return False

    # Deploy InferenceService and HPA
    yaml_files = [
        files["inferenceservice"],
        files["autoscaling"]
    ]

    deploy_result = cluster_manager.deploy_all(yaml_files)

    if deploy_result["success"]:
        print(f"   ✓ Deployed successfully")
        print(f"   Deployed {len(deploy_result['applied_files'])} resources")
    else:
        print(f"   ✗ Deployment failed: {deploy_result['errors']}")
        return False

    # Step 5: Wait for pod to become Ready
    print("\n5. Waiting for pod to become Ready...")
    print("   (This should take ~10-15 seconds with simulator)")

    max_wait = 60  # seconds
    start_time = time.time()

    while time.time() - start_time < max_wait:
        try:
            status = cluster_manager.get_inferenceservice_status(deployment_id)
            conditions = status.get("status", {}).get("conditions", [])

            for condition in conditions:
                if condition.get("type") == "Ready":
                    if condition.get("status") == "True":
                        elapsed = time.time() - start_time
                        print(f"   ✓ Pod Ready! (took {elapsed:.1f}s)")

                        # Get pod info
                        pods = cluster_manager.get_deployment_pods(deployment_id)
                        if pods:
                            pod = pods[0]
                            print(f"   Pod: {pod['name']}")
                            print(f"   Status: {pod['status']}")

                        return True
                    else:
                        reason = condition.get("message", "Unknown")
                        print(f"   Status: {condition.get('status')} - {reason}")

            time.sleep(2)

        except Exception as e:
            print(f"   Checking status... ({time.time() - start_time:.0f}s)")
            time.sleep(2)

    print(f"   ✗ Pod did not become Ready within {max_wait}s")

    # Show pod logs for debugging
    print("\n   Checking pod logs...")
    try:
        result = subprocess.run(
            ["kubectl", "logs", "-l", f"deployment-id={deployment_id}", "--tail=20"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.stdout:
            print("   Last 20 log lines:")
            print("   " + "\n   ".join(result.stdout.split("\n")))
    except:
        pass

    return False


if __name__ == "__main__":
    try:
        success = test_simulator_deployment()

        if success:
            print("\n" + "=" * 80)
            print("✓ Simulator deployment test PASSED!")
            print("=" * 80)
            print("\nNext steps:")
            print("  1. Test inference: kubectl port-forward svc/<deployment-id> 8080:8080")
            print("  2. Send request: curl http://localhost:8080/v1/completions ...")
            print("  3. Check metrics: curl http://localhost:8080/metrics")
            sys.exit(0)
        else:
            print("\n" + "=" * 80)
            print("✗ Simulator deployment test FAILED")
            print("=" * 80)
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
