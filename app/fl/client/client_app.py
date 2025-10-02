"""
Federated Learning client for surgical outcome optimization.
"""
import os
import warnings
from typing import Dict, Tuple, Optional
import logging

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context
from flwr.common.typing import NDArrays, Scalar
from omegaconf import DictConfig
from transformers import TrainingArguments
from trl import SFTTrainer

from app.fl.client.dataset import (
    get_tokenizer_and_data_collator,
    load_hospital_data,
    formatting_prompts_func
)
from app.fl.client.models import (
    cosine_annealing,
    get_model,
    set_parameters,
    get_parameters,
)
from app.utils.logger import app_logger

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
warnings.filterwarnings("ignore", category=UserWarning)


class SurgicalFLClient(NumPyClient):
    """Federated learning client for surgical outcome optimization."""

    def __init__(
        self,
        hospital_id: str,
        model_cfg: DictConfig,
        train_cfg: DictConfig,
        num_rounds: int,
        device: Optional[str] = None,
    ):
        """Initialize the FL client."""
        self.hospital_id = hospital_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.num_rounds = num_rounds
        
        app_logger.info(f"Initializing FL client for hospital {hospital_id}")
        app_logger.info(f"Using device: {self.device}")
        
        # Initialize model
        self.model = get_model(model_cfg, device_map=self.device)
        
        # Initialize tokenizer and data collator
        self.tokenizer, self.data_collator = get_tokenizer_and_data_collator(
            model_cfg.name
        )
        
        # Load hospital data
        self.trainset = self._load_data()
        
        # Setup training arguments
        self.training_arguments = self._setup_training_arguments()
        
        app_logger.info(f"Client initialized with {len(self.trainset['instruction'])} training samples")

    def _load_data(self) -> Dict:
        """Load hospital-specific training data."""
        return load_hospital_data(
            hospital_id=self.hospital_id,
            num_partitions=1,  # Each hospital is one partition
            min_samples=10
        )
    
    def _setup_training_arguments(self) -> TrainingArguments:
        """Setup training arguments."""
        return TrainingArguments(
            output_dir=f"./data/hospitals/hospital_{self.hospital_id}/checkpoints",
            per_device_train_batch_size=self.train_cfg.training_arguments.per_device_train_batch_size,
            gradient_accumulation_steps=self.train_cfg.training_arguments.gradient_accumulation_steps,
            num_train_epochs=self.train_cfg.training_arguments.num_train_epochs,
            logging_steps=self.train_cfg.training_arguments.logging_steps,
            save_strategy=self.train_cfg.training_arguments.save_strategy,
            evaluation_strategy=self.train_cfg.training_arguments.evaluation_strategy,
            fp16=self.train_cfg.training_arguments.fp16 and self.device != "cpu",
            gradient_checkpointing=self.train_cfg.training_arguments.gradient_checkpointing,
            report_to="none",
            remove_unused_columns=False,
            push_to_hub=False,
            load_best_model_at_end=False,
            metric_for_best_model=None,
            greater_is_better=None,
            warmup_steps=10,
            logging_dir=f"./logs/hospital_{self.hospital_id}",
            disable_tqdm=True,
        )

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict]:
        """Train the model on local data."""
        app_logger.info(f"Starting training round {config.get('current_round', 0)} for hospital {self.hospital_id}")
        
        # Set model parameters from server
        set_parameters(self.model, parameters)
        
        # Update learning rate based on round
        current_round = int(config.get("current_round", 0))
        new_lr = cosine_annealing(
            current_round,
            self.num_rounds,
            self.train_cfg.learning_rate_max,
            self.train_cfg.learning_rate_min,
        )
        self.training_arguments.learning_rate = new_lr
        
        app_logger.info(f"Learning rate for round {current_round}: {new_lr:.6f}")
        
        # Create trainer
        trainer = SFTTrainer(
            model=self.model,
            tokenizer=self.tokenizer,
            args=self.training_arguments,
            max_seq_length=self.train_cfg.seq_length,
            train_dataset=self.trainset,
            formatting_func=formatting_prompts_func,
            data_collator=self.data_collator,
            packing=False,
        )
        
        # Train
        try:
            results = trainer.train()
            
            # Get metrics
            metrics = {
                "train_loss": results.training_loss if results.training_loss else 0.0,
                "hospital_id": self.hospital_id,
                "round": current_round,
                "learning_rate": new_lr,
            }
            
            app_logger.info(f"Training completed. Loss: {metrics['train_loss']:.4f}")
            
        except Exception as e:
            app_logger.error(f"Training failed: {e}")
            metrics = {"train_loss": float('inf'), "error": str(e)}
        
        # Return updated parameters
        updated_parameters = get_parameters(self.model)
        num_examples = len(self.trainset['instruction'])
        
        return updated_parameters, num_examples, metrics

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict]:
        """Evaluate the model (optional)."""
        # For now, return dummy metrics
        # In production, you'd evaluate on a validation set
        return 0.0, len(self.trainset['instruction']), {"status": "evaluated"}


def client_fn(context: Context) -> SurgicalFLClient:
    """Create a Flower client for a hospital."""
    
    # Get configuration from context
    hospital_id = context.node_config.get("hospital_id", "default")
    num_rounds = context.run_config.get("num-server-rounds", 10)
    
    # Load configurations
    from omegaconf import OmegaConf
    config_path = "app/fl/config/model_config.yaml"
    cfg = OmegaConf.load(config_path)
    
    # Create and return client
    return SurgicalFLClient(
        hospital_id=hospital_id,
        model_cfg=cfg.model,
        train_cfg=cfg.train,
        num_rounds=num_rounds,
    ).to_client()


# Create Flower ClientApp
app = ClientApp(client_fn)