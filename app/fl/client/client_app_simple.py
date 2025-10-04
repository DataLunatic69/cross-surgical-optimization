"""
Simplified Federated Learning client compatible with flwr 1.6.0.
"""
import os
import warnings
from typing import Dict, Tuple, List
import numpy as np

from flwr.client import NumPyClient
from flwr.common.typing import NDArrays, Scalar

from app.utils.logger import app_logger

warnings.filterwarnings("ignore", category=UserWarning)


class SimpleSurgicalFLClient(NumPyClient):
    """Simplified FL client for testing without actual model training."""

    def __init__(self, hospital_id: str = "1", num_rounds: int = 10):
        self.hospital_id = hospital_id
        self.num_rounds = num_rounds
        self.current_round = 0
        app_logger.info(f"Initializing simplified FL client for hospital {hospital_id}")
        
        # Mock training data
        self.trainset = {"instruction": ["mock"] * 100, "input": ["mock"] * 100, "response": ["mock"] * 100}
        self.num_parameters = 1000
        self.parameters = self._initialize_parameters()

    def _initialize_parameters(self) -> NDArrays:
        """Initialize mock parameters."""
        return [np.random.randn(100, 10).astype(np.float32) for _ in range(10)]

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        """Return the current parameters."""
        app_logger.info(f"Hospital {self.hospital_id}: Getting parameters")
        return self.parameters

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        """Simulate training on local data."""
        self.current_round = int(config.get("current_round", 0))
        app_logger.info(f"Hospital {self.hospital_id}: Round {self.current_round}")
        
        # Simulate training by slightly modifying parameters
        updated_parameters = []
        for param in parameters:
            noise = np.random.randn(*param.shape).astype(np.float32) * 0.01
            updated_parameters.append(param + noise)
        
        # Store updated parameters
        self.parameters = updated_parameters
        
        # Simulate decreasing loss over rounds
        simulated_loss = max(0.1, 2.0 - (0.2 * self.current_round) + np.random.uniform(-0.1, 0.1))
        
        metrics = {
            "train_loss": float(simulated_loss),
            "hospital_id": self.hospital_id,
            "round": self.current_round
        }
        
        return updated_parameters, len(self.trainset["instruction"]), metrics

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict]:
        """Simulate evaluation."""
        eval_loss = max(0.1, 2.0 - (0.15 * self.current_round))
        return float(eval_loss), len(self.trainset["instruction"]), {
            "eval_loss": float(eval_loss),
            "hospital_id": self.hospital_id
        }


# For standalone testing
if __name__ == "__main__":
    # Test the client
    client = SimpleSurgicalFLClient(hospital_id="test", num_rounds=5)
    
    # Test get_parameters
    params = client.get_parameters({})
    print(f"Got {len(params)} parameter arrays")
    
    # Test fit
    config = {"current_round": 1}
    updated_params, num_samples, metrics = client.fit(params, config)
    print(f"Fit completed: {num_samples} samples, loss: {metrics['train_loss']:.4f}")
    
    # Test evaluate
    eval_loss, eval_samples, eval_metrics = client.evaluate(updated_params, config)
    print(f"Evaluation: loss: {eval_loss:.4f}")