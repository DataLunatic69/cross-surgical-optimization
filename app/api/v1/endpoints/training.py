"""
API endpoints for managing federated learning training sessions.
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, status
from sqlalchemy.orm import Session

from app.db.base import get_db
from app.services.training import fl_orchestration_service
from app.models.training import TrainingSession, TrainingStatus
from app.models.user import User, UserRole
from app.core.security import get_current_user
from app.utils.logger import app_logger
from pydantic import BaseModel


router = APIRouter()


class TrainingRequest(BaseModel):
    """Request model for starting training."""
    num_rounds: int = 10
    min_clients: int = 2
    hospital_ids: Optional[List[int]] = None


class TrainingResponse(BaseModel):
    """Response model for training session."""
    id: int
    session_name: str
    status: str
    num_rounds: int
    current_round: Optional[int]
    min_clients: int


@router.post("/start", response_model=TrainingResponse)
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_user),  # Add this line
    db: Session = Depends(get_db)
):
    """Start a new federated learning training session."""
    # Check if user has permission to start training
    if current_user.role not in [UserRole.SUPER_ADMIN, UserRole.HOSPITAL_ADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions to start training"
        )
    try:
        # Start training in background
        session = await fl_orchestration_service.start_training(
            num_rounds=request.num_rounds,
            min_clients=request.min_clients,
            hospitals=request.hospital_ids
        )
        
        return TrainingResponse(
            id=session.id,
            session_name=session.session_name,
            status=session.status.value,
            num_rounds=session.num_rounds,
            current_round=session.current_round,
            min_clients=session.min_clients
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        app_logger.error(f"Error starting training: {e}")
        raise HTTPException(status_code=500, detail="Failed to start training")


@router.post("/{session_id}/stop")
async def stop_training(session_id: int):
    """Stop an active training session."""
    success = await fl_orchestration_service.stop_training(session_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Training session not found or not active")
    
    return {"message": f"Training session {session_id} stopped successfully"}


@router.get("/{session_id}/status")
async def get_training_status(session_id: int):
    """Get the status of a training session."""
    status = fl_orchestration_service.get_session_status(session_id)
    
    if not status:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    return status


@router.get("/active", response_model=List[dict])
async def list_active_sessions():
    """List all active training sessions."""
    return fl_orchestration_service.list_active_sessions()


@router.get("/history", response_model=List[dict])
async def get_training_history(
    limit: int = 10,
    db: Session = Depends(get_db)
):
    """Get training session history."""
    sessions = db.query(TrainingSession)\
        .order_by(TrainingSession.created_at.desc())\
        .limit(limit)\
        .all()
    
    return [
        {
            "id": s.id,
            "session_name": s.session_name,
            "status": s.status.value,
            "num_rounds": s.num_rounds,
            "current_round": s.current_round,
            "started_at": s.started_at,
            "completed_at": s.completed_at,
            "global_model_loss": s.global_model_loss,
        }
        for s in sessions
    ]


@router.get("/{session_id}/metrics")
async def get_training_metrics(
    session_id: int,
    db: Session = Depends(get_db)
):
    """Get detailed metrics for a training session."""
    session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Training session not found")
    
    # Get round metrics
    rounds = []
    for round in session.rounds:
        rounds.append({
            "round_number": round.round_number,
            "num_participants": round.num_participants,
            "avg_loss": round.avg_loss,
            "started_at": round.started_at,
            "completed_at": round.completed_at,
        })
    
    return {
        "session_id": session.id,
        "session_name": session.session_name,
        "status": session.status.value,
        "rounds": rounds,
        "global_metrics": {
            "final_loss": session.global_model_loss,
            "total_rounds": session.num_rounds,
            "completed_rounds": session.current_round or 0,
        }
    }