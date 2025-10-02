"""
Federated learning training session models.
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, JSON, Text, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
import enum
from app.db.base import Base


class TrainingStatus(enum.Enum):
    """Training session status enumeration."""
    PENDING = "pending"
    INITIALIZING = "initializing"
    IN_PROGRESS = "in_progress"
    AGGREGATING = "aggregating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TrainingSession(Base):
    """Federated learning training session."""
    
    __tablename__ = "training_sessions"
    __table_args__ = {"schema": "training"}
    
    id = Column(Integer, primary_key=True, index=True)
    session_name = Column(String(255), unique=True, nullable=False)
    description = Column(Text)
    
    # Training configuration
    num_rounds = Column(Integer, nullable=False)
    min_clients = Column(Integer, nullable=False)
    fraction_fit = Column(Float, default=1.0)
    fraction_eval = Column(Float, default=1.0)
    
    # Model configuration
    base_model = Column(String(255), nullable=False)
    model_version = Column(String(50))
    model_config = Column(JSON)  # LoRA config, quantization, etc.
    
    # Status tracking
    status = Column(SQLEnum(TrainingStatus), default=TrainingStatus.PENDING, nullable=False)
    current_round = Column(Integer, default=0)
    
    # Performance metrics
    global_model_loss = Column(Float)
    global_model_accuracy = Column(Float)
    best_round = Column(Integer)
    best_loss = Column(Float)
    
    # Resource tracking
    total_computation_time = Column(Float)  # seconds
    total_communication_cost = Column(Float)  # MB
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    rounds = relationship("TrainingRound", back_populates="session", cascade="all, delete-orphan")
    participants = relationship("TrainingParticipant", back_populates="session")
    contributions = relationship("HospitalContribution", back_populates="training_session")
    model_versions = relationship("ModelVersion", back_populates="training_session")
    
    def __repr__(self):
        return f"<TrainingSession(name={self.session_name}, status={self.status})>"


class TrainingRound(Base):
    """Individual round in a training session."""
    
    __tablename__ = "training_rounds"
    __table_args__ = {"schema": "training"}
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("training.training_sessions.id"), nullable=False)
    round_number = Column(Integer, nullable=False)
    
    # Participation
    num_participants = Column(Integer)
    selected_clients = Column(JSON)  # List of hospital IDs
    
    # Metrics
    avg_loss = Column(Float)
    avg_accuracy = Column(Float)
    min_loss = Column(Float)
    max_loss = Column(Float)
    std_loss = Column(Float)
    
    # Aggregation details
    aggregation_method = Column(String(50), default="FedAvg")
    aggregation_weights = Column(JSON)  # Weights used for aggregation
    
    # Timing
    started_at = Column(DateTime(timezone=True))
    completed_at = Column(DateTime(timezone=True))
    duration_seconds = Column(Float)
    
    # Relationships
    session = relationship("TrainingSession", back_populates="rounds")
    
    def __repr__(self):
        return f"<TrainingRound(session={self.session_id}, round={self.round_number})>"


class TrainingParticipant(Base):
    """Hospital participation in training sessions."""
    
    __tablename__ = "training_participants"
    __table_args__ = {"schema": "training"}
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("training.training_sessions.id"), nullable=False)
    hospital_id = Column(Integer, ForeignKey("hospital.hospitals.id"), nullable=False)
    
    # Participation details
    joined_round = Column(Integer, default=1)
    left_round = Column(Integer)
    total_rounds_participated = Column(Integer, default=0)
    
    # Performance
    avg_loss = Column(Float)
    avg_accuracy = Column(Float)
    total_samples_contributed = Column(Integer)
    
    # Resource usage
    total_computation_time = Column(Float)
    total_communication_mb = Column(Float)
    
    # Status
    is_active = Column(Boolean, default=True)
    disconnection_count = Column(Integer, default=0)
    
    # Timestamps
    joined_at = Column(DateTime(timezone=True), server_default=func.now())
    last_active = Column(DateTime(timezone=True))
    
    # Relationships
    session = relationship("TrainingSession", back_populates="participants")
    hospital = relationship("Hospital", back_populates="training_participants")
    
    def __repr__(self):
        return f"<TrainingParticipant(session={self.session_id}, hospital={self.hospital_id})>"


class ModelVersion(Base):
    """Track model versions and checkpoints."""
    
    __tablename__ = "model_versions"
    __table_args__ = {"schema": "training"}
    
    id = Column(Integer, primary_key=True, index=True)
    version_tag = Column(String(100), unique=True, nullable=False)
    training_session_id = Column(Integer, ForeignKey("training.training_sessions.id"))
    
    # Model details
    base_model = Column(String(255), nullable=False)
    model_path = Column(String(500), nullable=False)  # File path to saved model
    model_size_mb = Column(Float)
    
    # Configuration
    model_config = Column(JSON)  # Complete model configuration
    training_config = Column(JSON)  # Training hyperparameters
    
    # Performance metrics
    test_loss = Column(Float)
    test_accuracy = Column(Float)
    validation_metrics = Column(JSON)
    
    # Deployment status
    is_deployed = Column(Boolean, default=False)
    deployed_at = Column(DateTime(timezone=True))
    deployment_endpoints = Column(JSON)  # List of endpoints using this model
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    training_session = relationship("TrainingSession", back_populates="model_versions")
    predictions = relationship("Prediction", back_populates="model_version")
    
    def __repr__(self):
        return f"<ModelVersion(tag={self.version_tag}, deployed={self.is_deployed})>"