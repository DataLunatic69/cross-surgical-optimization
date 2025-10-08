"""
User and authentication related models.
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Enum as SQLEnum
from sqlalchemy.orm import relationship, Session
from sqlalchemy.sql import func
from datetime import datetime, timedelta
from typing import List
import enum
import json
from app.db.base import Base
from passlib.context import CryptContext

# Password context for hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

class UserRole(enum.Enum):
    """User role enumeration."""
    SUPER_ADMIN = "super_admin"
    HOSPITAL_ADMIN = "hospital_admin"
    CLINICIAN = "clinician"
    RESEARCHER = "researcher"
    FL_CLIENT = "fl_client"


class User(Base):
    """User model for authentication and authorization."""
    
    __tablename__ = "users"
    __table_args__ = {"schema": "auth"}
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    username = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255))
    role = Column(SQLEnum(UserRole), nullable=False, default=UserRole.CLINICIAN)
    
    # Hospital association (nullable for super admins)
    hospital_id = Column(Integer, ForeignKey("hospital.hospitals.id"), nullable=True)
    
    # Account status
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_login = Column(DateTime(timezone=True))
    
    # Relationships
    hospital = relationship("Hospital", back_populates="users")
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="user")
    def verify_password(self, plain_password: str) -> bool:
        """Verify password against hash."""
        return pwd_context.verify(plain_password, self.hashed_password)
    
    @classmethod
    def get_password_hash(cls, password: str) -> str:
        """Create password hash."""
        # Ensure password is within bcrypt's 72 byte limit
        password_bytes = password.encode('utf-8')
        if len(password_bytes) > 72:
            password = password_bytes[:72].decode('utf-8', errors='ignore')
        return pwd_context.hash(password)
    
    def update_last_login(self, db: Session):
        """Update last login timestamp."""
        self.last_login = datetime.now()
        db.commit()
    
    def create_api_key(self, db: Session, name: str, description: str = None, scopes: List[str] = None):
        """Create a new API key for the user."""
        import secrets
        key = secrets.token_urlsafe(32)
        
        api_key = APIKey(
            key=key,
            name=name,
            description=description,
            user_id=self.id,
            scopes=json.dumps(scopes) if scopes else "[]",
            expires_at=datetime.now() + timedelta(days=365)  # 1 year expiry
        )
        db.add(api_key)
        db.commit()
        db.refresh(api_key)
        return api_key
    def __repr__(self):
        return f"<User(username={self.username}, email={self.email}, role={self.role})>"


class APIKey(Base):
    """API Key model for FL client authentication."""
    
    __tablename__ = "api_keys"
    __table_args__ = {"schema": "auth"}
    
    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    description = Column(String(500))
    
    # Owner
    user_id = Column(Integer, ForeignKey("auth.users.id"), nullable=False)
    
    # Permissions
    scopes = Column(String(500))  # JSON array of allowed scopes
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    last_used = Column(DateTime(timezone=True))
    expires_at = Column(DateTime(timezone=True))
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="api_keys")
    
    def __repr__(self):
        return f"<APIKey(name={self.name}, user_id={self.user_id})>"


class UserSession(Base):
    """User session model for tracking active sessions."""
    
    __tablename__ = "user_sessions"
    __table_args__ = {"schema": "auth"}
    
    id = Column(Integer, primary_key=True, index=True)
    session_token = Column(String(255), unique=True, index=True, nullable=False)
    refresh_token = Column(String(255), unique=True, index=True)
    
    # User association
    user_id = Column(Integer, ForeignKey("auth.users.id"), nullable=False)
    
    # Session metadata
    ip_address = Column(String(45))
    user_agent = Column(String(500))
    
    # Status
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    expires_at = Column(DateTime(timezone=True), nullable=False)
    last_activity = Column(DateTime(timezone=True), server_default=func.now())
    
    # Relationships
    user = relationship("User", back_populates="sessions")
    
    def __repr__(self):
        return f"<UserSession(user_id={self.user_id}, is_active={self.is_active})>"