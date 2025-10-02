"""
Dataset handling for surgical federated learning.
"""
import json
import os
from typing import Dict, List, Optional, Any
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from app.utils.logger import app_logger
from app.fl.common.data_structures import (
    SurgicalCase, 
    TrainingDataPoint, 
    create_surgical_prompt,
    SurgeryType
)


def get_tokenizer_and_data_collator(model_name: str):
    """Get tokenizer and data collator for the model."""
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        use_fast=True, 
        padding_side="right",
        trust_remote_code=True
    )
    
    # Set padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Response template for surgical recommendations
    response_template = "\nRecommended Surgical Approach:"
    response_template_ids = tokenizer.encode(
        response_template, add_special_tokens=False
    )[2:]  # Skip first tokens
    
    data_collator = DataCollatorForCompletionOnlyLM(
        response_template_ids, 
        tokenizer=tokenizer
    )
    
    return tokenizer, data_collator


def formatting_prompts_func(example: Dict[str, Any]) -> List[str]:
    """Format prompts for training."""
    output_texts = []
    
    # System message for medical context
    system_msg = (
        "You are an expert surgical consultant AI. Provide detailed, "
        "evidence-based surgical recommendations based on patient profiles "
        "and procedural requirements."
    )
    
    for i in range(len(example["instruction"])):
        text = f"""{system_msg}

### Instruction:
{example['instruction'][i]}

### Input:
{example['input'][i]}

### Response:
{example['response'][i]}"""
        
        output_texts.append(text)
    
    return output_texts


class SurgicalDatasetLoader:
    """Load and prepare surgical data for federated learning."""
    
    def __init__(self, hospital_id: str, data_dir: str = "./data/hospitals"):
        self.hospital_id = hospital_id
        self.data_dir = Path(data_dir) / f"hospital_{hospital_id}"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
    def load_surgical_cases(self) -> List[SurgicalCase]:
        """Load surgical cases from hospital data."""
        cases_file = self.data_dir / "surgical_cases.json"
        
        if cases_file.exists():
            with open(cases_file, 'r') as f:
                cases_data = json.load(f)
                return [SurgicalCase(**case) for case in cases_data]
        else:
            app_logger.warning(f"No surgical cases found for hospital {self.hospital_id}")
            return []
    
    def prepare_training_data(self, cases: List[SurgicalCase]) -> List[TrainingDataPoint]:
        """Convert surgical cases to training data points."""
        training_data = []
        
        for case in cases:
            try:
                data_point = create_surgical_prompt(case)
                training_data.append(data_point)
            except Exception as e:
                app_logger.error(f"Error processing case {case.case_id}: {e}")
                continue
        
        return training_data
    
    def create_dataset_dict(self, training_data: List[TrainingDataPoint]) -> Dict:
        """Create dataset dictionary for training."""
        return {
            "instruction": [dp.instruction for dp in training_data],
            "input": [dp.input for dp in training_data],
            "response": [dp.response for dp in training_data]
        }
    
    def save_processed_data(self, training_data: List[TrainingDataPoint]):
        """Save processed training data."""
        output_file = self.data_dir / "processed_training_data.json"
        
        data_dicts = [dp.dict() for dp in training_data]
        with open(output_file, 'w') as f:
            json.dump(data_dicts, f, indent=2, default=str)
        
        app_logger.info(f"Saved {len(data_dicts)} training examples to {output_file}")


def load_hospital_data(
    hospital_id: str, 
    num_partitions: int, 
    min_samples: int = 10
) -> Dict:
    """Load data for a specific hospital in federated learning."""
    
    loader = SurgicalDatasetLoader(hospital_id)
    
    # Load surgical cases
    cases = loader.load_surgical_cases()
    
    if len(cases) < min_samples:
        app_logger.warning(
            f"Hospital {hospital_id} has only {len(cases)} cases, "
            f"minimum {min_samples} required. Generating synthetic data..."
        )
        cases.extend(generate_synthetic_cases(min_samples - len(cases)))
    
    # Convert to training data
    training_data = loader.prepare_training_data(cases)
    
    # Save processed data
    loader.save_processed_data(training_data)
    
    # Return dataset dictionary
    return loader.create_dataset_dict(training_data)


def generate_synthetic_cases(num_cases: int) -> List[SurgicalCase]:
    """Generate synthetic surgical cases for testing."""
    import random
    
    synthetic_cases = []
    
    surgery_types = list(SurgeryType)
    procedures = {
        SurgeryType.CARDIAC: ["CABG", "Valve Replacement", "Angioplasty"],
        SurgeryType.ORTHOPEDIC: ["Hip Replacement", "Knee Arthroscopy", "Spinal Fusion"],
        SurgeryType.NEUROLOGICAL: ["Craniotomy", "Spinal Tumor Removal", "DBS"],
        SurgeryType.GENERAL: ["Appendectomy", "Cholecystectomy", "Hernia Repair"],
    }
    
    for i in range(num_cases):
        surgery_type = random.choice(surgery_types[:4])  # Use main types
        
        case = SurgicalCase(
            case_id=f"synthetic_{i:04d}",
            patient_profile={
                "age": random.randint(25, 75),
                "gender": random.choice(["male", "female"]),
                "bmi": round(random.uniform(18.5, 35.0), 1),
                "medical_history": random.sample(
                    ["Hypertension", "Diabetes", "None"], 
                    k=random.randint(0, 2)
                ),
                "current_medications": random.sample(
                    ["Aspirin", "Metformin", "Lisinopril", "None"],
                    k=random.randint(0, 2)
                ),
                "allergies": random.sample(["Penicillin", "None"], k=1),
                "lab_results": {
                    "Hemoglobin": round(random.uniform(10, 16), 1),
                    "WBC": round(random.uniform(4, 11), 1),
                    "Platelets": random.randint(150, 400)
                },
                "vital_signs": {
                    "BP_systolic": random.randint(110, 140),
                    "BP_diastolic": random.randint(70, 90),
                    "Heart_rate": random.randint(60, 100)
                },
                "risk_factors": random.sample(
                    ["Obesity", "Smoking", "Age", "None"],
                    k=random.randint(0, 2)
                )
            },
            surgery_type=surgery_type,
            procedure_name=random.choice(procedures[surgery_type]),
            urgency=random.choice(["elective", "urgent"]),
            surgeon_experience_years=random.randint(5, 25),
            hospital_volume=random.choice(["high", "medium", "low"]),
            equipment_available=["Standard OR Equipment", "Monitoring Systems"],
            technique_used=random.choice(["Standard", "Minimally Invasive"]),
            duration_hours=round(random.uniform(1, 6), 1),
            blood_loss_ml=random.randint(50, 500),
            complications=random.sample(
                ["None", "Minor bleeding", "Infection"], 
                k=random.randint(0, 1)
            ),
            success=random.random() > 0.1,  # 90% success rate
            recovery_days=random.randint(3, 14),
            readmission_30_days=random.random() < 0.1,  # 10% readmission
            recommended_approach=f"Standard {surgery_type.value} approach with careful monitoring",
            key_considerations=[
                "Patient stability",
                "Surgical site preparation",
                "Post-operative monitoring"
            ],
            risk_mitigation=[
                "Prophylactic antibiotics",
                "DVT prophylaxis",
                "Early mobilization"
            ]
        )
        
        synthetic_cases.append(case)
    
    return synthetic_cases