"""
Custom federated learning strategy for surgical optimization.
"""
from typing import List, Tuple, Union, Optional, Dict
from io import BytesIO

from flwr.common import (
    FitIns, 
    FitRes, 
    Parameters, 
    Scalar,
    parameters_to_ndarrays,
    ndarrays_to_parameters
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from app.utils.logger import app_logger
from app.db.base import SessionLocal
from app.models.training import TrainingSession, TrainingStatus


class SurgicalFLStrategy(FedAvg):
    """Custom FL strategy for surgical outcome optimization."""
    
    def __init__(self, fl_manager=None, **kwargs):
        """Initialize the strategy."""
        super().__init__(**kwargs)
        self.fl_manager = fl_manager
        self.round_metrics = {}
        self.communication_cost_mb = 0.0
        
    def configure_fit(
        self, 
        server_round: int, 
        parameters: Parameters, 
        client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        
        app_logger.info(f"Configuring round {server_round}")
        
        # Update training session status
        if self.fl_manager:
            with SessionLocal() as db:
                session = db.query(TrainingSession).get(self.fl_manager.training_session_id)
                if session:
                    session.status = TrainingStatus.IN_PROGRESS
                    session.current_round = server_round
                    db.commit()
        
        # Get client configuration
        config = {}
        if self.on_fit_config_fn:
            config = self.on_fit_config_fn(server_round)
        
        config["current_round"] = server_round
        
        # Configure fit
        fit_ins = FitIns(parameters, config)
        
        # Sample clients
        sample_size = self.num_fit_clients(client_manager.num_available())
        clients = client_manager.sample(
            num_clients=sample_size, 
            min_num_clients=self.min_fit_clients
        )
        
        app_logger.info(f"Selected {len(clients)} clients for round {server_round}")
        
        # Track communication cost
        self._track_communication_cost(parameters, len(clients), "download")
        
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        
        if failures:
            app_logger.warning(f"Round {server_round} had {len(failures)} failures")
        
        if not results:
            app_logger.error(f"Round {server_round}: No results to aggregate")
            return None, {}
        
        # Track upload communication cost
        for _, fit_res in results:
            self._track_communication_cost(fit_res.parameters, 1, "upload")
        
        # Aggregate parameters
        parameters_aggregated, metrics_aggregated = super().aggregate_fit(
            server_round, results, failures
        )
        
        # Store round metrics
        self.round_metrics[server_round] = metrics_aggregated
        
        # Update database with round results
        if self.fl_manager and metrics_aggregated:
            with SessionLocal() as db:
                self.fl_manager.update_training_round(
                    db, server_round, metrics_aggregated
                )
        
        # Log metrics
        app_logger.info(f"Round {server_round} metrics: {metrics_aggregated}")
        app_logger.info(f"Total communication cost: {self.communication_cost_mb:.2f} MB")
        
        return parameters_aggregated, metrics_aggregated
    
    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure evaluation round."""
        
        # For now, we'll skip client evaluation
        # In production, you'd evaluate on validation data
        return []
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        
        # Call parent's evaluate function if defined
        if self.evaluate_fn is not None:
            loss, metrics = self.evaluate_fn(server_round, self.current_parameters, {})
            return loss, metrics
        
        return None, {}
    
    def _track_communication_cost(
        self, 
        parameters: Parameters, 
        num_clients: int,
        direction: str
    ):
        """Track communication costs in MB."""
        
        # Calculate size of parameters
        size_bytes = sum(
            BytesIO(tensor).getbuffer().nbytes 
            for tensor in parameters.tensors
        )
        
        size_mb = (size_bytes * num_clients) / (1024 * 1024)
        self.communication_cost_mb += size_mb
        
        app_logger.debug(f"Communication {direction}: {size_mb:.2f} MB")
        
        # Warn if exceeding limits
        if self.communication_cost_mb > 200000:  # 200 GB
            app_logger.warning(
                f"High communication cost: {self.communication_cost_mb:.2f} MB"
            )
    
    def __del__(self):
        """Cleanup when strategy is destroyed."""
        
        # Update final training session status
        if hasattr(self, 'fl_manager') and self.fl_manager:
            with SessionLocal() as db:
                session = db.query(TrainingSession).get(
                    self.fl_manager.training_session_id
                )
                if session:
                    session.status = TrainingStatus.COMPLETED
                    session.completed_at = datetime.now()
                    
                    # Save final metrics
                    if self.round_metrics:
                        last_round = max(self.round_metrics.keys())
                        last_metrics = self.round_metrics[last_round]
                        session.global_model_loss = last_metrics.get("avg_loss", 0.0)
                    
                    db.commit()
                    app_logger.info("Training session completed and saved to database")