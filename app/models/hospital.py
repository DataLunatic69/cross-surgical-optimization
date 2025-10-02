"""
Hospital management models for federated learning participants.
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Float, JSON, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base import Base


class Hospital(Base):
    """Hospital model representing FL participants."""
    
    __tablename__ = "hospitals"
    __table_args__ = {"schema": "hospital"}
    
    id = Column(Integer, primary_key=True, index=True)
    code = Column(String(50), unique=True, index=True, nullable=False)
    name = Column(String(255), nullable=False)
    
    # Contact information
    address = Column(Text)
    city = Column(String(100))
    state = Column(String(100))
    country = Column(String(100))
    postal_code = Column(String(20))
    phone = Column(String(50))
    email = Column(String(255))
    
    # Hospital metadata
    type = Column(String(100))  # e.g., "General", "Specialty", "Teaching"
    bed_capacity = Column(Integer)
    annual_surgeries = Column(Integer)
    specialties = Column(JSON)  # List of surgical specialties
    
    # FL Configuration
    fl_client_id = Column(String(255), unique=True, index=True)
    fl_certificate = Column(Text)  # SSL certificate for secure communication
    data_contribution_score = Column(Float, default=0.0)
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    is_approved = Column(Boolean, default=False, nullable=False)
    approval_date = Column(DateTime(timezone=True))
    
    # Performance metrics
    total_training_rounds = Column(Integer, default=0)
    total_predictions = Column(Integer, default=0)
    average_model_performance = Column(Float)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_participation = Column(DateTime(timezone=True))
    
    # Relationships
    users = relationship("User", back_populates="hospital")
    training_participants = relationship("TrainingParticipant", back_populates="hospital")
    contributions = relationship("HospitalContribution", back_populates="hospital")
    
    def __repr__(self):
        return f"<Hospital(code={self.code}, name={self.name})>"


class HospitalContribution(Base):
    """Track hospital contributions to federated learning."""
    
    __tablename__ = "hospital_contributions"
    __table_args__ = {"schema": "hospital"}
    
    id = Column(Integer, primary_key=True, index=True)
    hospital_id = Column(Integer, ForeignKey("hospital.hospitals.id"), nullable=False)
    
    # Contribution metrics
    training_session_id = Column(Integer, ForeignKey("training.training_sessions.id"))
    round_number = Column(Integer)
    
    # Data statistics
    num_samples = Column(Integer)
    num_surgeries = Column(Integer)
    data_quality_score = Column(Float)
    
    # Model metrics
    local_loss = Column(Float)
    local_accuracy = Column(Float)
    computation_time = Column(Float)  # in seconds
    
    # Communication metrics
    upload_size_mb = Column(Float)
    download_size_mb = Column(Float)
    
    # Timestamps
    contributed_at = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    hospital = relationship("Hospital", back_populates="contributions")
    training_session = relationship("TrainingSession", back_populates="contributions")
    
    def __repr__(self):
        return f"<HospitalContribution(hospital_id={self.hospital_id}, session={self.training_session_id})>"


class HospitalConfig(Base):
    """Hospital-specific configuration for FL participation."""
    
    __tablename__ = "hospital_configs"
    __table_args__ = {"schema": "hospital"}
    
    id = Column(Integer, primary_key=True, index=True)
    hospital_id = Column(Integer, ForeignKey("hospital.hospitals.id"), unique=True, nullable=False)
    
    # Data configuration
    min_samples_for_training = Column(Integer, default=100)
    data_retention_days = Column(Integer, default=90)
    
    # Privacy settings
    differential_privacy_enabled = Column(Boolean, default=True)
    epsilon = Column(Float, default=1.0)  # DP epsilon parameter
    delta = Column(Float, default=1e-5)   # DP delta parameter
    
    # Resource limits
    max_computation_time = Column(Integer, default=3600)  # seconds
    max_memory_usage_gb = Column(Float, default=16.0)
    max_upload_size_mb = Column(Float, default=100.0)
    
    # Training preferences
    preferred_batch_size = Column(Integer, default=32)
    preferred_learning_rate = Column(Float, default=0.001)
    local_epochs = Column(Integer, default=5)
    
    # Scheduling
    participation_schedule = Column(JSON)  # Days/hours when hospital can participate
    timezone = Column(String(50), default="UTC")
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    def __repr__(self):
        return f"<HospitalConfig(hospital_id={self.hospital_id})>"