"""
Prediction and surgical outcome models.
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, JSON, Text, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from app.db.base import Base


class SurgeryType(enum.Enum):
    """Types of surgical procedures."""
    CARDIAC = "cardiac"
    ORTHOPEDIC = "orthopedic"
    NEUROLOGICAL = "neurological"
    GENERAL = "general"
    PEDIATRIC = "pediatric"
    ONCOLOGICAL = "oncological"
    TRANSPLANT = "transplant"
    EMERGENCY = "emergency"
    MINIMALLY_INVASIVE = "minimally_invasive"
    OTHER = "other"


class RiskLevel(enum.Enum):
    """Risk level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Prediction(Base):
    """Surgical outcome predictions."""
    
    __tablename__ = "predictions"
    __table_args__ = {"schema": "prediction"}
    
    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(String(100), unique=True, index=True, nullable=False)
    
    # User and model
    user_id = Column(Integer, ForeignKey("auth.users.id"), nullable=False)
    model_version_id = Column(Integer, ForeignKey("training.model_versions.id"), nullable=False)
    
    # Patient profile (anonymized)
    patient_age = Column(Integer)
    patient_gender = Column(String(20))
    patient_bmi = Column(Float)
    medical_history = Column(JSON)  # List of conditions
    current_medications = Column(JSON)  # List of medications
    lab_results = Column(JSON)  # Key lab values
    
    # Surgery details
    surgery_type = Column(SQLEnum(SurgeryType), nullable=False)
    procedure_name = Column(String(255))
    urgency_level = Column(String(50))  # elective, urgent, emergency
    
    # Prediction inputs
    surgeon_experience_years = Column(Integer)
    hospital_volume_category = Column(String(50))  # high, medium, low
    equipment_availability = Column(JSON)
    
    # Prediction outputs
    recommended_approach = Column(Text)  # Detailed surgical approach
    success_probability = Column(Float)
    risk_level = Column(SQLEnum(RiskLevel))
    estimated_duration_hours = Column(Float)
    estimated_recovery_days = Column(Integer)
    
    # Detailed recommendations
    pre_operative_recommendations = Column(JSON)
    intra_operative_considerations = Column(JSON)
    post_operative_care_plan = Column(JSON)
    potential_complications = Column(JSON)
    
    # Risk factors
    identified_risk_factors = Column(JSON)
    mitigation_strategies = Column(JSON)
    
    # Confidence and explanation
    confidence_score = Column(Float)
    explanation = Column(Text)
    alternative_approaches = Column(JSON)
    
    # Performance tracking
    inference_time_ms = Column(Float)
    
    # Feedback (if available)
    actual_outcome = Column(JSON)
    feedback_received = Column(Boolean, default=False)
    outcome_matched = Column(Boolean)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    feedback_at = Column(DateTime(timezone=True))
    
    # Relationships
    user = relationship("User", back_populates="predictions")
    model_version = relationship("ModelVersion", back_populates="predictions")
    
    def __repr__(self):
        return f"<Prediction(id={self.prediction_id}, surgery={self.surgery_type})>"


class SurgicalOutcome(Base):
    """Historical surgical outcomes for training and validation."""
    
    __tablename__ = "surgical_outcomes"
    __table_args__ = {"schema": "prediction"}
    
    id = Column(Integer, primary_key=True, index=True)
    hospital_id = Column(Integer, ForeignKey("hospital.hospitals.id"), nullable=False)
    
    # Case identifier (anonymized)
    case_id = Column(String(100), unique=True, index=True)
    
    # Patient data (anonymized)
    patient_age_group = Column(String(20))  # e.g., "20-30", "30-40"
    patient_risk_factors = Column(JSON)
    
    # Surgery information
    surgery_type = Column(SQLEnum(SurgeryType))
    procedure_details = Column(JSON)
    surgery_date = Column(DateTime(timezone=True))
    
    # Surgical team
    surgeon_experience_level = Column(String(50))  # junior, mid, senior, expert
    team_size = Column(Integer)
    
    # Outcomes
    success = Column(Boolean)
    complications = Column(JSON)
    recovery_time_days = Column(Integer)
    readmission_30_days = Column(Boolean)
    mortality_30_days = Column(Boolean)
    
    # Quality metrics
    blood_loss_ml = Column(Float)
    surgery_duration_minutes = Column(Float)
    icu_days = Column(Integer)
    total_hospital_days = Column(Integer)
    
    # Cost metrics (optional)
    relative_cost_index = Column(Float)
    
    # Used for training
    used_in_training = Column(Boolean, default=False)
    training_session_id = Column(Integer, ForeignKey("training.training_sessions.id"))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<SurgicalOutcome(case={self.case_id}, success={self.success})>"


class PredictionFeedback(Base):
    """Feedback on predictions for continuous improvement."""
    
    __tablename__ = "prediction_feedback"
    __table_args__ = {"schema": "prediction"}
    
    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("prediction.predictions.id"), nullable=False)
    
    # Feedback provider
    provided_by_user_id = Column(Integer, ForeignKey("auth.users.id"))
    provider_role = Column(String(50))  # surgeon, clinician, researcher
    
    # Accuracy feedback
    accuracy_rating = Column(Integer)  # 1-5 scale
    usefulness_rating = Column(Integer)  # 1-5 scale
    
    # Outcome data
    actual_surgery_duration = Column(Float)
    actual_recovery_days = Column(Integer)
    actual_complications = Column(JSON)
    surgery_success = Column(Boolean)
    
    # Qualitative feedback
    comments = Column(Text)
    suggested_improvements = Column(Text)
    
    # Flags
    flag_for_review = Column(Boolean, default=False)
    incorporated_in_training = Column(Boolean, default=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    def __repr__(self):
        return f"<PredictionFeedback(prediction={self.prediction_id}, rating={self.accuracy_rating})>"