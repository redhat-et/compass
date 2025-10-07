"""FastAPI routes for the Pre-Deployment Assistant API."""

import logging
from datetime import datetime
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..context_intent.schema import DeploymentRecommendation, ConversationMessage
from ..orchestration.workflow import RecommendationWorkflow
from ..knowledge_base.model_catalog import ModelCatalog
from ..knowledge_base.slo_templates import SLOTemplateRepository
from ..deployment.generator import DeploymentGenerator
from ..deployment.validator import YAMLValidator, ValidationError
from ..deployment.cluster import KubernetesClusterManager, KubernetesDeploymentError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="AI Pre-Deployment Assistant API",
    description="API for LLM deployment recommendations",
    version="0.1.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize workflow (singleton for POC)
workflow = RecommendationWorkflow()
model_catalog = ModelCatalog()
slo_repo = SLOTemplateRepository()
# Use simulator mode by default (no GPU required for development)
deployment_generator = DeploymentGenerator(simulator_mode=True)
yaml_validator = YAMLValidator()

# Initialize cluster manager (will be None if cluster not accessible)
try:
    cluster_manager = KubernetesClusterManager(namespace="default")
    logger.info("Kubernetes cluster manager initialized successfully")
except KubernetesDeploymentError as e:
    logger.warning(f"Kubernetes cluster not accessible: {e}")
    cluster_manager = None


# Request/Response models
class RecommendationRequest(BaseModel):
    """Request for deployment recommendation."""
    user_message: str
    conversation_history: Optional[List[ConversationMessage]] = None


class SimpleRecommendationRequest(BaseModel):
    """Simple request for deployment recommendation (UI compatibility)."""
    message: str


class RecommendationResponse(BaseModel):
    """Response with deployment recommendation."""
    recommendation: DeploymentRecommendation
    success: bool = True
    message: Optional[str] = None


class ErrorResponse(BaseModel):
    """Error response."""
    success: bool = False
    error: str
    details: Optional[str] = None


class DeploymentRequest(BaseModel):
    """Request to generate deployment YAML files."""
    recommendation: DeploymentRecommendation
    namespace: str = "default"


class DeploymentResponse(BaseModel):
    """Response with generated deployment files."""
    deployment_id: str
    namespace: str
    files: dict
    success: bool = True
    message: Optional[str] = None


class DeploymentStatusResponse(BaseModel):
    """Mock deployment status response."""
    deployment_id: str
    status: str
    slo_compliance: dict
    resource_utilization: dict
    cost_analysis: dict
    traffic_patterns: dict
    recommendations: Optional[List[str]] = None


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "ai-pre-deployment-assistant"}


# Main recommendation endpoint
@app.post("/api/v1/recommend", response_model=RecommendationResponse)
async def get_recommendation(request: RecommendationRequest):
    """
    Generate deployment recommendation from user message.

    Args:
        request: Recommendation request with user message

    Returns:
        Deployment recommendation

    Raises:
        HTTPException: If recommendation fails
    """
    try:
        logger.info(f"Received recommendation request: {request.user_message[:100]}...")

        recommendation = workflow.generate_recommendation(
            user_message=request.user_message,
            conversation_history=request.conversation_history
        )

        # Validate recommendation
        is_valid = workflow.validate_recommendation(recommendation)
        if not is_valid:
            logger.warning("Generated recommendation failed validation")

        return RecommendationResponse(
            recommendation=recommendation,
            success=True,
            message="Recommendation generated successfully"
        )

    except Exception as e:
        logger.error(f"Failed to generate recommendation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate recommendation: {str(e)}"
        )


# Get available models
@app.get("/api/v1/models")
async def list_models():
    """Get list of available models."""
    try:
        models = model_catalog.get_all_models()
        return {
            "models": [model.to_dict() for model in models],
            "count": len(models)
        }
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Get GPU types
@app.get("/api/v1/gpu-types")
async def list_gpu_types():
    """Get list of available GPU types."""
    try:
        gpu_types = model_catalog.get_all_gpu_types()
        return {
            "gpu_types": [gpu.to_dict() for gpu in gpu_types],
            "count": len(gpu_types)
        }
    except Exception as e:
        logger.error(f"Failed to list GPU types: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Get use case templates
@app.get("/api/v1/use-cases")
async def list_use_cases():
    """Get list of supported use cases with SLO templates."""
    try:
        templates = slo_repo.get_all_templates()
        return {
            "use_cases": {
                use_case: template.to_dict()
                for use_case, template in templates.items()
            },
            "count": len(templates)
        }
    except Exception as e:
        logger.error(f"Failed to list use cases: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Simplified recommendation endpoint for UI
@app.post("/api/recommend")
async def simple_recommend(request: SimpleRecommendationRequest):
    """
    Simplified recommendation endpoint for UI compatibility.

    Args:
        request: Simple request with message field

    Returns:
        Recommendation as JSON dict
    """
    try:
        logger.info(f"Received UI recommendation request: {request.message[:100]}...")

        recommendation = workflow.generate_recommendation(
            user_message=request.message,
            conversation_history=None
        )

        # Return recommendation as dict for easier UI consumption
        return recommendation.model_dump()

    except Exception as e:
        logger.error(f"Failed to generate recommendation: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate recommendation: {str(e)}"
        )


# Simple test endpoint for quick validation
@app.post("/api/v1/test")
async def test_endpoint(message: str = "I need a chatbot for 1000 users"):
    """
    Quick test endpoint for validation.

    Args:
        message: Test message (optional)

    Returns:
        Simplified recommendation
    """
    try:
        recommendation = workflow.generate_recommendation(message)

        return {
            "success": True,
            "model": recommendation.model_name,
            "gpu_config": f"{recommendation.gpu_config.gpu_count}x {recommendation.gpu_config.gpu_type}",
            "cost_per_month": f"${recommendation.cost_per_month_usd:.2f}",
            "meets_slo": recommendation.meets_slo,
            "reasoning": recommendation.reasoning
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# Deployment endpoints
@app.post("/api/deploy", response_model=DeploymentResponse)
async def deploy_model(request: DeploymentRequest):
    """
    Generate deployment YAML files from recommendation.

    Args:
        request: Deployment request with recommendation

    Returns:
        Deployment response with file paths

    Raises:
        HTTPException: If deployment generation fails
    """
    try:
        logger.info(f"Generating deployment for model: {request.recommendation.model_name}")

        # Generate all YAML files
        result = deployment_generator.generate_all(
            recommendation=request.recommendation,
            namespace=request.namespace
        )

        # Validate generated files
        try:
            yaml_validator.validate_all(result["files"])
            logger.info(f"All YAML files validated for deployment: {result['deployment_id']}")
        except ValidationError as e:
            logger.error(f"YAML validation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Generated YAML validation failed: {str(e)}"
            )

        return DeploymentResponse(
            deployment_id=result["deployment_id"],
            namespace=result["namespace"],
            files=result["files"],
            success=True,
            message=f"Deployment files generated successfully for {result['deployment_id']}"
        )

    except Exception as e:
        logger.error(f"Failed to generate deployment: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate deployment: {str(e)}"
        )


@app.get("/api/deployments/{deployment_id}/status", response_model=DeploymentStatusResponse)
async def get_deployment_status(deployment_id: str):
    """
    Get mock deployment status for observability demonstration.

    This endpoint returns simulated observability data to demonstrate
    Component 9 (Inference Observability & SLO Monitoring).

    Args:
        deployment_id: Deployment identifier

    Returns:
        Mock deployment status with observability metrics

    Raises:
        HTTPException: If deployment not found
    """
    try:
        logger.info(f"Fetching mock status for deployment: {deployment_id}")

        # Mock observability data (in production, this would query Prometheus/Grafana)
        import random

        # Simulate SLO compliance with some variance
        base_ttft = 185
        base_tpot = 48
        base_e2e = 1850

        mock_status = DeploymentStatusResponse(
            deployment_id=deployment_id,
            status="running",
            slo_compliance={
                "ttft_p90_ms": base_ttft + random.randint(-10, 15),
                "ttft_target_ms": 200,
                "ttft_compliant": True,
                "tpot_p90_ms": base_tpot + random.randint(-3, 5),
                "tpot_target_ms": 50,
                "tpot_compliant": True,
                "e2e_p95_ms": base_e2e + random.randint(-50, 100),
                "e2e_target_ms": 2000,
                "e2e_compliant": True,
                "throughput_qps": 122 + random.randint(-5, 10),
                "throughput_target_qps": 100,
                "throughput_compliant": True,
                "uptime_pct": 99.94 + random.uniform(-0.05, 0.05),
                "uptime_target_pct": 99.9,
                "uptime_compliant": True
            },
            resource_utilization={
                "gpu_utilization_pct": 78 + random.randint(-5, 10),
                "gpu_memory_used_gb": 14.2 + random.uniform(-1, 2),
                "gpu_memory_total_gb": 24,
                "avg_batch_size": 18 + random.randint(-3, 5),
                "queue_depth": random.randint(0, 5),
                "token_throughput_per_gpu": 3500 + random.randint(-200, 300)
            },
            cost_analysis={
                "actual_cost_per_hour_usd": 0.95 + random.uniform(-0.05, 0.1),
                "predicted_cost_per_hour_usd": 1.00,
                "actual_cost_per_month_usd": 812 + random.randint(-30, 50),
                "predicted_cost_per_month_usd": 800,
                "cost_per_1k_tokens_usd": 0.042 + random.uniform(-0.002, 0.005),
                "predicted_cost_per_1k_tokens_usd": 0.040
            },
            traffic_patterns={
                "avg_prompt_tokens": 165 + random.randint(-10, 20),
                "predicted_prompt_tokens": 150,
                "avg_generation_tokens": 220 + random.randint(-15, 25),
                "predicted_generation_tokens": 200,
                "peak_qps": 95 + random.randint(-5, 10),
                "predicted_peak_qps": 100,
                "requests_last_hour": 7200 + random.randint(-500, 800),
                "requests_last_24h": 172800 + random.randint(-5000, 10000)
            },
            recommendations=[
                "GPU utilization is 78%, below the 80% efficiency target. Consider downsizing to reduce cost.",
                "Actual traffic is 10% higher than predicted. Monitor for potential capacity constraints.",
                "All SLO targets are being met with headroom. Configuration is performing well."
            ]
        )

        return mock_status

    except Exception as e:
        logger.error(f"Failed to get deployment status: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Deployment not found: {deployment_id}"
        )


@app.post("/api/deploy-to-cluster")
async def deploy_to_cluster(request: DeploymentRequest):
    """
    Deploy model to Kubernetes cluster.

    This endpoint generates YAML files AND applies them to the cluster.

    Args:
        request: Deployment request with recommendation and namespace

    Returns:
        Deployment result with status

    Raises:
        HTTPException: If cluster not accessible or deployment fails
    """
    # Try to initialize cluster manager if it wasn't available at startup
    manager = cluster_manager
    if manager is None:
        try:
            manager = KubernetesClusterManager(namespace=request.namespace)
        except KubernetesDeploymentError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Kubernetes cluster not accessible: {str(e)}"
            )

    try:
        logger.info(f"Deploying model to cluster: {request.recommendation.model_name}")

        # Step 1: Generate YAML files
        result = deployment_generator.generate_all(
            recommendation=request.recommendation,
            namespace=request.namespace
        )

        deployment_id = result["deployment_id"]
        files = result["files"]

        # Step 2: Validate generated files
        try:
            yaml_validator.validate_all(files)
            logger.info(f"YAML validation passed for: {deployment_id}")
        except ValidationError as e:
            logger.error(f"YAML validation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Generated YAML validation failed: {str(e)}"
            )

        # Step 3: Deploy to cluster
        # Note: generator creates keys like "inferenceservice", "vllm-config", "autoscaling", etc.
        # Skip ServiceMonitor for now (requires Prometheus Operator)
        yaml_file_paths = [
            files["inferenceservice"],
            files["autoscaling"]
        ]

        deployment_result = manager.deploy_all(yaml_file_paths)

        if not deployment_result["success"]:
            logger.error(f"Deployment failed: {deployment_result['errors']}")
            raise HTTPException(
                status_code=500,
                detail=f"Deployment failed: {deployment_result['errors']}"
            )

        logger.info(f"Successfully deployed {deployment_id} to cluster")

        return {
            "success": True,
            "deployment_id": deployment_id,
            "namespace": request.namespace,
            "files": files,
            "deployment_result": deployment_result,
            "message": f"Successfully deployed {deployment_id} to Kubernetes cluster"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to deploy to cluster: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to deploy to cluster: {str(e)}"
        )


@app.get("/api/cluster-status")
async def get_cluster_status():
    """
    Get Kubernetes cluster status.

    Returns:
        Cluster accessibility and basic info
    """
    # Always try to check cluster status dynamically, don't rely on cached cluster_manager
    try:
        temp_manager = KubernetesClusterManager(namespace="default")
        # List existing deployments
        deployments = temp_manager.list_inferenceservices()

        return {
            "accessible": True,
            "namespace": temp_manager.namespace,
            "inference_services": deployments,
            "count": len(deployments),
            "message": "Cluster accessible"
        }
    except Exception as e:
        logger.error(f"Failed to query cluster status: {e}")
        return {
            "accessible": False,
            "error": str(e)
        }


@app.get("/api/deployments/{deployment_id}/k8s-status")
async def get_k8s_deployment_status(deployment_id: str):
    """
    Get actual Kubernetes deployment status (not mock data).

    Args:
        deployment_id: InferenceService name

    Returns:
        Real deployment status from cluster

    Raises:
        HTTPException: If cluster not accessible
    """
    # Try to initialize cluster manager if it wasn't available at startup
    manager = cluster_manager
    if manager is None:
        try:
            manager = KubernetesClusterManager(namespace="default")
        except KubernetesDeploymentError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Kubernetes cluster not accessible: {str(e)}"
            )

    try:
        # Get InferenceService status
        isvc_status = manager.get_inferenceservice_status(deployment_id)

        # Get associated pods
        pods = manager.get_deployment_pods(deployment_id)

        return {
            "deployment_id": deployment_id,
            "inferenceservice": isvc_status,
            "pods": pods,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Failed to get K8s deployment status: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get deployment status: {str(e)}"
        )


@app.delete("/api/deployments/{deployment_id}")
async def delete_deployment(deployment_id: str):
    """
    Delete a deployment from the cluster.

    Args:
        deployment_id: InferenceService name to delete

    Returns:
        Deletion result

    Raises:
        HTTPException: If cluster not accessible or deletion fails
    """
    # Try to initialize cluster manager if it wasn't available at startup
    manager = cluster_manager
    if manager is None:
        try:
            manager = KubernetesClusterManager(namespace="default")
        except KubernetesDeploymentError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Kubernetes cluster not accessible: {str(e)}"
            )

    try:
        result = manager.delete_inferenceservice(deployment_id)

        if not result["success"]:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to delete deployment: {result.get('error', 'Unknown error')}"
            )

        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete deployment: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete deployment: {str(e)}"
        )


@app.get("/api/deployments")
async def list_all_deployments():
    """
    List all InferenceServices in the cluster with their detailed status.

    Returns:
        List of deployments with status information

    Raises:
        HTTPException: If cluster not accessible
    """
    # Try to initialize cluster manager if it wasn't available at startup
    manager = cluster_manager
    if manager is None:
        try:
            manager = KubernetesClusterManager(namespace="default")
        except KubernetesDeploymentError as e:
            raise HTTPException(
                status_code=503,
                detail=f"Kubernetes cluster not accessible: {str(e)}"
            )

    try:
        # Get list of all InferenceService names
        deployment_ids = manager.list_inferenceservices()

        # Get detailed status for each
        deployments = []
        for deployment_id in deployment_ids:
            status = manager.get_inferenceservice_status(deployment_id)
            pods = manager.get_deployment_pods(deployment_id)

            deployments.append({
                "deployment_id": deployment_id,
                "status": status,
                "pods": pods
            })

        return {
            "success": True,
            "count": len(deployments),
            "deployments": deployments,
            "namespace": manager.namespace
        }

    except Exception as e:
        logger.error(f"Failed to list deployments: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list deployments: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
