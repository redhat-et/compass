"""API route modules for NeuralNav."""

from .configuration import router as configuration_router
from .health import router as health_router
from .intent import router as intent_router
from .recommendation import router as recommendation_router
from .reference_data import router as reference_data_router
from .specification import router as specification_router

__all__ = [
    "health_router",
    "intent_router",
    "specification_router",
    "recommendation_router",
    "configuration_router",
    "reference_data_router",
]
