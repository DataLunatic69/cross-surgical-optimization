"""
Model registry - Import all models to ensure they're registered with SQLAlchemy.
"""
from app.models.user import User, APIKey, UserSession, UserRole
from app.models.hospital import Hospital, HospitalContribution, HospitalConfig
from app.models.training import (
    TrainingSession,
    TrainingRound,
    TrainingParticipant,
    ModelVersion,
    TrainingStatus
)
from app.models.prediction import (
    Prediction,
    SurgicalOutcome,
    PredictionFeedback,
    SurgeryType,
    RiskLevel
)

# Export all models
__all__ = [
    # User models
    "User",
    "APIKey", 
    "UserSession",
    "UserRole",
    
    # Hospital models
    "Hospital",
    "HospitalContribution",
    "HospitalConfig",
    
    # Training models
    "TrainingSession",
    "TrainingRound",
    "TrainingParticipant",
    "ModelVersion",
    "TrainingStatus",
    
    # Prediction models
    "Prediction",
    "SurgicalOutcome",
    "PredictionFeedback",
    "SurgeryType",
    "RiskLevel"
]