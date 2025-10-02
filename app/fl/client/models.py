"""
Model management for surgical federated learning.
"""
import math
import torch
from omegaconf import DictConfig
from collections import OrderedDict
from typing import Optional, Dict, Any
from peft import (
    LoraConfig,
    get_peft_model,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training
)
from transformers import (
    AutoModelForCausalLM, 
    BitsAndBytesConfig,
    AutoConfig
)
from flwr.common.typing import NDArrays
from app.utils.logger import app_logger


def cosine_annealing(
    current_round: int,
    total_round: int,
    lrate_max: float = 0.001,
    lrate_min: float = 0.0,
) -> float:
    """Implement cosine annealing learning rate schedule."""
    cos_inner = math.pi * current_round / total_round
    return lrate_min + 0.5 * (lrate_max - lrate_min) * (1 + math.cos(cos_inner))


def get_model(model_cfg: DictConfig, device_map: str = "auto"):
    """Load model with appropriate quantization config and optimizations."""
    
    app_logger.info(f"Loading model: {model_cfg.name}")
    
    # Configure quantization
    quantization_config = None
    if model_cfg.quantization == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
    elif model_cfg.quantization == 8:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_compute_dtype=torch.bfloat16
        )
    
    # Load model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg.name,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            cache_dir=model_cfg.get("cache_dir", "./data/models"),
            low_cpu_mem_usage=True
        )
    except Exception as e:
        app_logger.error(f"Error loading model {model_cfg.name}: {e}")
        app_logger.info("Falling back to smaller model...")
        
        # Fallback to a smaller model for testing
        model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",  # Smaller model for testing
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    
    # Prepare for training
    if quantization_config:
        model = prepare_model_for_kbit_training(
            model, 
            use_gradient_checkpointing=model_cfg.gradient_checkpointing
        )
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=model_cfg.lora.peft_lora_r,
        lora_alpha=model_cfg.lora.peft_lora_alpha,
        target_modules=list(model_cfg.lora.target_modules),
        lora_dropout=model_cfg.lora.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    # Enable gradient checkpointing if specified
    if model_cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    
    # Get PEFT model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model


def set_parameters(model, parameters: NDArrays) -> None:
    """Change the parameters of the model using the given ones."""
    try:
        peft_state_dict_keys = get_peft_model_state_dict(model).keys()
        params_dict = zip(peft_state_dict_keys, parameters)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        set_peft_model_state_dict(model, state_dict)
        app_logger.debug("Model parameters updated successfully")
    except Exception as e:
        app_logger.error(f"Error setting model parameters: {e}")
        raise


def get_parameters(model) -> NDArrays:
    """Return the parameters of the current model."""
    try:
        state_dict = get_peft_model_state_dict(model)
        return [val.cpu().numpy() for _, val in state_dict.items()]
    except Exception as e:
        app_logger.error(f"Error getting model parameters: {e}")
        raise


class ModelManager:
    """Manage model loading, saving, and versioning."""
    
    def __init__(self, base_path: str = "./data/models"):
        self.base_path = base_path
        self.current_model = None
        self.model_config = None
    
    def load_base_model(self, model_cfg: DictConfig) -> Any:
        """Load the base model."""
        self.model_config = model_cfg
        self.current_model = get_model(model_cfg)
        return self.current_model
    
    def save_checkpoint(
        self, 
        model: Any, 
        round_num: int, 
        metrics: Optional[Dict[str, float]] = None
    ):
        """Save model checkpoint."""
        checkpoint_path = f"{self.base_path}/checkpoints/round_{round_num}"
        
        try:
            # Save PEFT model
            model.save_pretrained(checkpoint_path)
            
            # Save metrics if provided
            if metrics:
                import json
                with open(f"{checkpoint_path}/metrics.json", 'w') as f:
                    json.dump(metrics, f, indent=2)
            
            app_logger.info(f"Saved checkpoint to {checkpoint_path}")
        except Exception as e:
            app_logger.error(f"Error saving checkpoint: {e}")
    
    def load_checkpoint(self, round_num: int) -> Optional[Any]:
        """Load model from checkpoint."""
        checkpoint_path = f"{self.base_path}/checkpoints/round_{round_num}"
        
        try:
            if self.current_model is None:
                app_logger.error("Base model not loaded")
                return None
            
            # Load PEFT weights
            from peft import PeftModel
            model = PeftModel.from_pretrained(
                self.current_model.base_model.model,
                checkpoint_path
            )
            
            app_logger.info(f"Loaded checkpoint from {checkpoint_path}")
            return model
        except Exception as e:
            app_logger.error(f"Error loading checkpoint: {e}")
            return None
    
    def get_model_size_mb(self) -> float:
        """Get model size in MB."""
        if self.current_model is None:
            return 0.0
        
        param_size = 0
        for param in self.current_model.parameters():
            param_size += param.nelement() * param.element_size()
        
        buffer_size = 0
        for buffer in self.current_model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()
        
        size_mb = (param_size + buffer_size) / 1024 / 1024
        return size_mb