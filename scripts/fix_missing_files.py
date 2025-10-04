"""
Create missing files needed for the FL system.
"""
import os
from pathlib import Path

def create_fl_config():
    """Create FL configuration file."""
    config_dir = Path("app/fl/config")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    config_content = """model:
  name: "microsoft/phi-2"
  quantization: 4
  gradient_checkpointing: true
  cache_dir: "./data/models"
  
  lora:
    peft_lora_r: 16
    peft_lora_alpha: 32
    target_modules:
      - "q_proj"
      - "k_proj" 
      - "v_proj"
      - "o_proj"
    lora_dropout: 0.075
    task_type: "CAUSAL_LM"

train:
  learning_rate_max: 0.0002
  learning_rate_min: 0.00001
  save_every_round: 2
  seq_length: 512
  
  training_arguments:
    output_dir: "./data/checkpoints"
    per_device_train_batch_size: 4
    gradient_accumulation_steps: 4
    num_train_epochs: 1
    logging_steps: 10
    save_strategy: "epoch"
    evaluation_strategy: "no"
    fp16: false
    gradient_checkpointing: true
    report_to: "none"
    remove_unused_columns: false
    
strategy:
  fraction_fit: 1.0
  fraction_evaluate: 1.0
  min_fit_clients: 2
  min_evaluate_clients: 2
  min_available_clients: 2

static:
  dataset:
    name: "surgical_outcomes"
"""
    
    config_path = config_dir / "model_config.yaml"
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✓ Created: {config_path}")


def create_simple_client():
    """Create simplified FL client."""
    client_dir = Path("app/fl/client")
    client_dir.mkdir(parents=True, exist_ok=True)
    
    client_content = '''"""
Simplified Federated Learning client without TRL dependency.
"""
import os
import warnings
from typing import Dict, Tuple, Optional, List
import numpy as np

from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.typing import NDArrays, Scalar

from app.utils.logger import app_logger

warnings.filterwarnings("ignore", category=UserWarning)


class SimpleSurgicalFLClient(NumPyClient):
    """Simplified FL client for testing without actual model training."""

    def __init__(self, hospital_id: str, num_rounds: int = 10):
        self.hospital_id = hospital_id
        self.num_rounds = num_rounds
        self.current_round = 0
        app_logger.info(f"Initializing simplified FL client for hospital {hospital_id}")
        
        # Mock training data
        self.trainset = {"instruction": ["mock"] * 100, "input": ["mock"] * 100, "response": ["mock"] * 100}
        self.num_parameters = 1000
        self.parameters = self._initialize_parameters()

    def _initialize_parameters(self) -> NDArrays:
        return [np.random.randn(100, 10).astype(np.float32) for _ in range(10)]

    def fit(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[NDArrays, int, Dict]:
        self.current_round = int(config.get("current_round", 0))
        app_logger.info(f"Hospital {self.hospital_id}: Round {self.current_round}")
        
        # Simulate training
        updated_parameters = []
        for param in parameters:
            noise = np.random.randn(*param.shape).astype(np.float32) * 0.01
            updated_parameters.append(param + noise)
        
        simulated_loss = max(0.1, 2.0 - (0.2 * self.current_round) + np.random.uniform(-0.1, 0.1))
        
        return updated_parameters, len(self.trainset["instruction"]), {
            "train_loss": float(simulated_loss),
            "hospital_id": self.hospital_id,
            "round": self.current_round
        }

    def evaluate(self, parameters: NDArrays, config: Dict[str, Scalar]) -> Tuple[float, int, Dict]:
        eval_loss = max(0.1, 2.0 - (0.15 * self.current_round))
        return float(eval_loss), len(self.trainset["instruction"]), {"eval_loss": float(eval_loss)}


def client_fn(context: Context) -> SimpleSurgicalFLClient:
    hospital_id = context.node_config.get("hospital_id", "1")
    num_rounds = context.run_config.get("num-server-rounds", 10)
    return SimpleSurgicalFLClient(hospital_id=hospital_id, num_rounds=num_rounds).to_client()


app = ClientApp(client_fn)
'''
    
    client_path = client_dir / "client_app_simple.py"
    with open(client_path, 'w') as f:
        f.write(client_content)
    
    print(f"✓ Created: {client_path}")


def create_init_files():
    """Create __init__.py files for packages."""
    dirs = [
        "app/fl",
        "app/fl/client",
        "app/fl/server",
        "app/fl/common",
        "app/services",
        "app/api/v1/endpoints"
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        init_path = Path(dir_path) / "__init__.py"
        if not init_path.exists():
            init_path.write_text('"""Package initialization."""\n')
            print(f"✓ Created: {init_path}")


def main():
    print("=" * 50)
    print("Creating Missing Files")
    print("=" * 50)
    
    print("\n1. Creating FL configuration...")
    create_fl_config()
    
    print("\n2. Creating simplified FL client...")
    create_simple_client()
    
    print("\n3. Creating __init__ files...")
    create_init_files()
    
    print("\n" + "=" * 50)
    print("✅ All missing files created!")
    print("\nNext steps:")
    print("1. Run: python scripts/test_fl_simple.py")
    print("2. Start API: python -m uvicorn app.main:app --reload")


if __name__ == "__main__":
    main()