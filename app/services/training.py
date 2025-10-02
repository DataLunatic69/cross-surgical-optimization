"""
Service for orchestrating federated learning training sessions.
"""
import asyncio
import subprocess
import threading
from typing import Optional, Dict, List, Any
from datetime import datetime
from pathlib import Path

from app.utils.logger import app_logger
from app.db.base import SessionLocal
from app.models.training import TrainingSession, TrainingStatus
from app.models.hospital import Hospital


class FLOrchestrationService:
    """Orchestrate federated learning training sessions."""
    
    def __init__(self):
        self.active_sessions: Dict[int, subprocess.Popen] = {}
        self.session_threads: Dict[int, threading.Thread] = {}
        
    async def start_training(
        self,
        num_rounds: int = 10,
        min_clients: int = 2,
        hospitals: Optional[List[int]] = None
    ) -> TrainingSession:
        """Start a new federated learning training session."""
        
        # Create training session in database
        with SessionLocal() as db:
            # Get available hospitals
            if hospitals:
                available_hospitals = db.query(Hospital).filter(
                    Hospital.id.in_(hospitals),
                    Hospital.is_active == True,
                    Hospital.is_approved == True
                ).all()
            else:
                available_hospitals = db.query(Hospital).filter(
                    Hospital.is_active == True,
                    Hospital.is_approved == True
                ).all()
            
            if len(available_hospitals) < min_clients:
                raise ValueError(
                    f"Not enough hospitals available. "
                    f"Required: {min_clients}, Available: {len(available_hospitals)}"
                )
            
            # Create session
            session = TrainingSession(
                session_name=f"surgical_fl_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                description="Surgical outcome optimization training",
                num_rounds=num_rounds,
                min_clients=min_clients,
                fraction_fit=1.0,
                fraction_eval=0.5,
                base_model="medalpaca/medalpaca-7b",
                model_config={
                    "num_hospitals": len(available_hospitals),
                    "hospital_ids": [h.id for h in available_hospitals]
                },
                status=TrainingStatus.PENDING
            )
            db.add(session)
            db.commit()
            db.refresh(session)
            
            session_id = session.id
        
        # Start FL server in background
        server_thread = threading.Thread(
            target=self._run_fl_server,
            args=(session_id, num_rounds),
            daemon=True
        )
        server_thread.start()
        self.session_threads[session_id] = server_thread
        
        # Start FL clients for each hospital
        await asyncio.sleep(5)  # Give server time to start
        
        for hospital in available_hospitals:
            client_thread = threading.Thread(
                target=self._run_fl_client,
                args=(hospital.id, session_id),
                daemon=True
            )
            client_thread.start()
        
        app_logger.info(f"Started training session {session_id} with {len(available_hospitals)} hospitals")
        
        return session
    
    def _run_fl_server(self, session_id: int, num_rounds: int):
        """Run FL server in subprocess."""
        try:
            # Update status
            with SessionLocal() as db:
                session = db.query(TrainingSession).get(session_id)
                session.status = TrainingStatus.INITIALIZING
                session.started_at = datetime.now()
                db.commit()
            
            # Start server process
            cmd = [
                "python", "-m", "flwr.server",
                "--app", "app.fl.server.server_app:app",
                "--config", f"num-server-rounds={num_rounds}",
                "--insecure"  # For development only
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.active_sessions[session_id] = process
            
            # Monitor process
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                app_logger.info(f"FL server for session {session_id} completed successfully")
                status = TrainingStatus.COMPLETED
            else:
                app_logger.error(f"FL server for session {session_id} failed: {stderr}")
                status = TrainingStatus.FAILED
            
            # Update status
            with SessionLocal() as db:
                session = db.query(TrainingSession).get(session_id)
                session.status = status
                session.completed_at = datetime.now()
                db.commit()
                
        except Exception as e:
            app_logger.error(f"Error running FL server: {e}")
            with SessionLocal() as db:
                session = db.query(TrainingSession).get(session_id)
                session.status = TrainingStatus.FAILED
                db.commit()
    
    def _run_fl_client(self, hospital_id: int, session_id: int):
        """Run FL client for a hospital."""
        try:
            # Start client process
            cmd = [
                "python", "-m", "flwr.client",
                "--app", "app.fl.client.client_app:app",
                "--node-config", f"hospital_id={hospital_id}",
                "--server", "127.0.0.1:8080",
                "--insecure"
            ]
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Monitor process
            stdout, stderr = process.communicate()
            
            if process.returncode == 0:
                app_logger.info(f"FL client for hospital {hospital_id} completed successfully")
            else:
                app_logger.error(f"FL client for hospital {hospital_id} failed: {stderr}")
                
        except Exception as e:
            app_logger.error(f"Error running FL client for hospital {hospital_id}: {e}")
    
    async def stop_training(self, session_id: int) -> bool:
        """Stop an active training session."""
        
        if session_id in self.active_sessions:
            process = self.active_sessions[session_id]
            process.terminate()
            
            # Wait for process to end
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
            
            del self.active_sessions[session_id]
            
            # Update database
            with SessionLocal() as db:
                session = db.query(TrainingSession).get(session_id)
                session.status = TrainingStatus.CANCELLED
                db.commit()
            
            app_logger.info(f"Stopped training session {session_id}")
            return True
        
        return False
    
    def get_session_status(self, session_id: int) -> Optional[Dict[str, Any]]:
        """Get the status of a training session."""
        
        with SessionLocal() as db:
            session = db.query(TrainingSession).get(session_id)
            
            if not session:
                return None
            
            return {
                "id": session.id,
                "name": session.session_name,
                "status": session.status.value,
                "current_round": session.current_round,
                "total_rounds": session.num_rounds,
                "started_at": session.started_at,
                "completed_at": session.completed_at,
                "is_active": session_id in self.active_sessions
            }
    
    def list_active_sessions(self) -> List[Dict[str, Any]]:
        """List all active training sessions."""
        
        active_sessions = []
        
        with SessionLocal() as db:
            sessions = db.query(TrainingSession).filter(
                TrainingSession.status.in_([
                    TrainingStatus.PENDING,
                    TrainingStatus.INITIALIZING,
                    TrainingStatus.IN_PROGRESS
                ])
            ).all()
            
            for session in sessions:
                active_sessions.append({
                    "id": session.id,
                    "name": session.session_name,
                    "status": session.status.value,
                    "current_round": session.current_round,
                    "total_rounds": session.num_rounds,
                    "started_at": session.started_at
                })
        
        return active_sessions


# Create global orchestration service instance
fl_orchestration_service = FLOrchestrationService()