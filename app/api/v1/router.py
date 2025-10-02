"""
Main API router for v1 endpoints.
"""
from fastapi import APIRouter

# Import endpoint routers
from app.api.v1.endpoints import training

api_router = APIRouter()

# Include sub-routers
api_router.include_router(training.router, prefix="/training", tags=["Training"])

# Temporary test endpoint
@api_router.get("/test")
async def test_endpoint():
    """Test endpoint to verify API is working."""
    return {"message": "API v1 is operational", "status": "success"}