# app/api/v1/endpoints/auth.py
from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime
import json
from app.db.base import get_db
from app.models.user import User, UserRole, APIKey
from app.services.auth_service import auth_service, get_current_user, require_role
from app.utils.logger import app_logger

router = APIRouter()

# Pydantic models for request/response
class UserCreate(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: str
    role: UserRole = UserRole.CLINICIAN
    hospital_id: Optional[int] = None

class UserResponse(BaseModel):
    id: int
    email: str
    username: str
    full_name: str
    role: UserRole
    hospital_id: Optional[int]
    is_active: bool
    is_verified: bool

class Token(BaseModel):
    access_token: str
    token_type: str
    user: UserResponse

class APIKeyCreate(BaseModel):
    name: str
    description: Optional[str] = None
    scopes: Optional[List[str]] = None

class APIKeyResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    key: str  # Only returned on creation
    scopes: List[str]
    is_active: bool
    expires_at: datetime
    last_used: Optional[datetime]

@router.post("/register", response_model=UserResponse)
async def register_user(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """Register a new user."""
    # Check if user already exists
    existing_user = db.query(User).filter(
        (User.email == user_data.email) | (User.username == user_data.username)
    ).first()
    
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User with this email or username already exists"
        )
    
    # Create new user
    hashed_password = User.get_password_hash(user_data.password)
    user = User(
        email=user_data.email,
        username=user_data.username,
        hashed_password=hashed_password,
        full_name=user_data.full_name,
        role=user_data.role,
        hospital_id=user_data.hospital_id,
        is_active=True,
        is_verified=False  # Email verification can be added later
    )
    
    db.add(user)
    db.commit()
    db.refresh(user)
    
    app_logger.info(f"New user registered: {user.email} ({user.role.value})")
    
    return UserResponse(
        id=user.id,
        email=user.email,
        username=user.username,
        full_name=user.full_name,
        role=user.role,
        hospital_id=user.hospital_id,
        is_active=user.is_active,
        is_verified=user.is_verified
    )

@router.post("/login", response_model=Token)
async def login_user(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """Login user and return access token."""
    user = auth_service.authenticate_user(db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    
    # Update last login
    user.update_last_login(db)
    
    # Create access token
    access_token = auth_service.create_access_token(
        data={"user_id": user.id, "role": user.role.value}
    )
    
    user_response = UserResponse(
        id=user.id,
        email=user.email,
        username=user.username,
        full_name=user.full_name,
        role=user.role,
        hospital_id=user.hospital_id,
        is_active=user.is_active,
        is_verified=user.is_verified
    )
    
    app_logger.info(f"User logged in: {user.email}")
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user=user_response
    )

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """Get current user information."""
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        username=current_user.username,
        full_name=current_user.full_name,
        role=current_user.role,
        hospital_id=current_user.hospital_id,
        is_active=current_user.is_active,
        is_verified=current_user.is_verified
    )

@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    api_key_data: APIKeyCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new API key for the current user."""
    api_key = current_user.create_api_key(
        db=db,
        name=api_key_data.name,
        description=api_key_data.description,
        scopes=api_key_data.scopes
    )
    
    return APIKeyResponse(
        id=api_key.id,
        name=api_key.name,
        description=api_key.description,
        key=api_key.key,  # Only time the full key is returned
        scopes=json.loads(api_key.scopes) if api_key.scopes else [],
        is_active=api_key.is_active,
        expires_at=api_key.expires_at,
        last_used=api_key.last_used
    )

@router.get("/api-keys", response_model=List[APIKeyResponse])
async def list_api_keys(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """List all API keys for the current user."""
    api_keys = db.query(APIKey).filter(
        APIKey.user_id == current_user.id
    ).all()
    
    return [
        APIKeyResponse(
            id=key.id,
            name=key.name,
            description=key.description,
            key="***" + key.key[-4:] if key.key else "",  # Mask the key
            scopes=json.loads(key.scopes) if key.scopes else [],
            is_active=key.is_active,
            expires_at=key.expires_at,
            last_used=key.last_used
        )
        for key in api_keys
    ]

@router.delete("/api-keys/{api_key_id}")
async def revoke_api_key(
    api_key_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Revoke (delete) an API key."""
    api_key = db.query(APIKey).filter(
        APIKey.id == api_key_id,
        APIKey.user_id == current_user.id
    ).first()
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="API key not found"
        )
    
    db.delete(api_key)
    db.commit()
    
    return {"message": "API key revoked successfully"}

@router.post("/logout")
async def logout_user(
    current_user: User = Depends(get_current_user)
):
    """Logout user (client should discard token)."""
    # In a stateless JWT system, logout is handled client-side
    # We could implement a token blacklist if needed
    return {"message": "Successfully logged out"}