"""
Dependencies for API endpoints.
"""
from typing import Generator, Dict, Any
from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.core.security import get_current_user, get_current_active_user


def get_db_session() -> Generator[Session, None, None]:
    """Get database session."""
    return get_db()


def get_current_user_dependency() -> Dict[str, Any]:
    """Dependency to get current user."""
    return get_current_user


def get_current_active_user_dependency() -> Dict[str, Any]:
    """Dependency to get current active user."""
    return get_current_active_user


# Common dependency combinations
def get_db_with_user():
    """Get both database session and current user."""
    def dependency(
        db: Session = Depends(get_db),
        current_user: Dict[str, Any] = Depends(get_current_active_user)
    ):
        return db, current_user
    return dependency