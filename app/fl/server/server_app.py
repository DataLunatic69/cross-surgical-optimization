"""
Federated Learning server for surgical outcome optimization.
"""
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

from flwr.common import Context, ndarrays_to_parameters, Parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from omegaconf import OmegaConf, DictConfig

from app.fl.client.models import get_model, get_parameters, set_parameters
from app.fl.server.strategy import SurgicalFLStrategy
from app.utils.logger import app_logger
from app.db.base import SessionLocal
from app.models.training import TrainingSession, TrainingStatus, TrainingRound
from sqlalchemy.orm import Session


class FLServerManager:
    """Manage FL server operations and database interactions."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.training_session_id = None
        self.save_path = self._create_save_path()
        
    def _create_save_path(self) -> Path:
        """Create directory for saving models."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = Path(f"./data/models/training_{timestamp}")
        save_path.mkdir(parents=True, exist_ok=True)
        return save_path
    
    def create_training_session(self, db: Session, num_rounds: int) -> TrainingSession:
        """Create a new training session in the database."""
        session = TrainingSession(
            session_name=f"surgical_fl_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description="Surgical outcome optimization federated learning",
            num_rounds=num_rounds,
            min_clients=self.config.strategy.min_fit_clients,
            fraction_fit=self.config.strategy.fraction_fit,
            fraction_eval=self.config.strategy.fraction_evaluate,
            base_model=self.config.model.name,
            model_config={
                "quantization": self.config.model.quantization,
                "lora_config": OmegaConf.to_container(self.config.model.lora)
            },
            status=TrainingStatus.INITIALIZING
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        
        self.training_session_id = session.id
        app_logger.info(f"Created training session: {session.session_name}")
        return session
    
    def update_training_round(
        self, 
        db: Session, 
        round_num: int, 
        metrics: Dict
    ):
        """Update training round in database."""
        round_record = TrainingRound(
            session_id=self.training_session_id,
            round_number=round_num,
            num_participants=metrics.get("num_participants", 0),
            avg_loss=metrics.get("avg_loss", 0.0),
            avg_accuracy=metrics.get("avg_accuracy", 0.0),
            started_at=datetime.now()
        )
        db.add(round_record)
        db.commit()


def get_evaluate_fn(model_cfg: DictConfig, save_path: Path, fl_manager: FLServerManager):
    """Return an evaluation function for the strategy."""
    
    def evaluate(server_round: int, parameters: Parameters, config: Dict):
        """Evaluate global model and save checkpoints."""
        
        if server_round % model_cfg.get("save_every_round", 2) == 0:
            try:
                # Initialize model
                model = get_model(model_cfg)
                
                # Set parameters
                params_array = parameters.tensors
                set_parameters(model, params_array)
                
                # Save model checkpoint
                checkpoint_path = save_path / f"checkpoint_round_{server_round}"
                model.save_pretrained(checkpoint_path)
                
                app_logger.info(f"Saved model checkpoint to {checkpoint_path}")
                
                # Update database
                with SessionLocal() as db:
                    session = db.query(TrainingSession).get(fl_manager.training_session_id)
                    if session:
                        session.current_round = server_round
                        db.commit()
                
            except Exception as e:
                app_logger.error(f"Error in evaluation: {e}")
        
        return 0.0, {"round": server_round, "status": "evaluated"}
    
    return evaluate


def get_on_fit_config(save_path: Path):
    """Return a function to configure fit rounds."""
    
    def fit_config_fn(server_round: int):
        """Configure the fit round."""
        return {
            "current_round": server_round,
            "save_path": str(save_path),
        }
    
    return fit_config_fn


def fit_weighted_average(metrics: List[Tuple[int, Dict]]) -> Dict:
    """Aggregate metrics from multiple clients."""
    
    if not metrics:
        return {}
    
    # Calculate weighted average loss
    total_examples = sum(num_examples for num_examples, _ in metrics)
    
    if total_examples == 0:
        return {}
    
    weighted_losses = [
        num_examples * m.get("train_loss", 0.0) 
        for num_examples, m in metrics
    ]
    
    avg_loss = sum(weighted_losses) / total_examples
    
    # Collect hospital IDs
    hospital_ids = [
        m.get("hospital_id", "unknown") 
        for _, m in metrics
    ]
    
    return {
        "avg_loss": avg_loss,
        "num_participants": len(metrics),
        "total_examples": total_examples,
        "participating_hospitals": hospital_ids,
    }


def server_fn(context: Context):
    """Construct components for the FL server."""
    
    # Load configuration
    config_path = "app/fl/config/model_config.yaml"
    cfg = OmegaConf.load(config_path)
    
    # Create FL manager
    fl_manager = FLServerManager(cfg)
    
    # Get number of rounds
    num_rounds = context.run_config.get("num-server-rounds", 10)
    
    # Create training session in database
    with SessionLocal() as db:
        fl_manager.create_training_session(db, num_rounds)
    
    # Get initial model parameters
    app_logger.info("Initializing global model...")
    init_model = get_model(cfg.model)
    init_parameters = ndarrays_to_parameters(get_parameters(init_model))
    
    # Define strategy
    strategy = SurgicalFLStrategy(
        fraction_fit=cfg.strategy.fraction_fit,
        fraction_evaluate=cfg.strategy.fraction_evaluate,
        min_fit_clients=cfg.strategy.min_fit_clients,
        min_evaluate_clients=cfg.strategy.min_evaluate_clients,
        min_available_clients=cfg.strategy.min_available_clients,
        on_fit_config_fn=get_on_fit_config(fl_manager.save_path),
        fit_metrics_aggregation_fn=fit_weighted_average,
        evaluate_fn=get_evaluate_fn(cfg.model, fl_manager.save_path, fl_manager),
        initial_parameters=init_parameters,
        fl_manager=fl_manager,
    )
    
    # Create server config
    config = ServerConfig(
        num_rounds=num_rounds,
        round_timeout=600,  # 10 minutes timeout per round
    )
    
    app_logger.info(f"Starting FL server with {num_rounds} rounds")
    
    return ServerAppComponents(strategy=strategy, config=config)


# Create Flower ServerApp
app = ServerApp(server_fn=server_fn)