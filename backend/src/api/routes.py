"""FastAPI routes for the Pre-Deployment Assistant API."""

import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ..context_intent.schema import DeploymentRecommendation, ConversationMessage
from ..orchestration.workflow import RecommendationWorkflow
from ..knowledge_base.model_catalog import ModelCatalog
from ..knowledge_base.slo_templates import SLOTemplateRepository

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
