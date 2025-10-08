"""
Pydantic schemas for user authentication and management.
"""
from typing import Optional, Dict, Any
from pydantic import BaseModel, EmailStr, validator


class UserBase(BaseModel):
    """Base user schema."""
    email: EmailStr
    full_name: Optional[str] = None
    role: Optional[str] = "clinician"
    hospital_id: Optional[int] = None


class UserCreate(UserBase):
    """Schema for creating a new user."""
    password: str
    
    @validator('password')
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters long')
        return v


class UserLogin(BaseModel):
    """Schema for user login."""
    email: EmailStr
    password: str


class UserResponse(UserBase):
    """Schema for user response (without sensitive data)."""
    id: int
    is_active: bool
    is_verified: bool
    
    class Config:
        from_attributes = True


class Token(BaseModel):
    """Schema for JWT token response."""
    access_token: str
    token_type: str
    expires_in: int
    user: UserResponse


class TokenData(BaseModel):
    """Schema for token payload data."""
    user_id: Optional[int] = None
    email: Optional[str] = None
    role: Optional[str] = None
    hospital_id: Optional[int] = None


class APIKeyCreate(BaseModel):
    """Schema for creating API key."""
    name: str
    description: Optional[str] = None
    scopes: Optional[list] = []


class APIKeyResponse(BaseModel):
    """Schema for API key response."""
    id: int
    name: str
    key: str
    scopes: list
    is_active: bool
    created_at: str
    expires_at: Optional[str] = None
    
    class Config:
        from_attributes = True


class PasswordChange(BaseModel):
    """Schema for password change."""
    current_password: str
    new_password: str
    
    @validator('new_password')
    def new_password_strength(cls, v):
        if len(v) < 8:
            raise ValueError('New password must be at least 8 characters long')
        return v