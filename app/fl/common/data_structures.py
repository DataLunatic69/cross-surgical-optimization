"""
Data structures for surgical federated learning.
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


class SurgeryType(str, Enum):
    """Types of surgical procedures."""
    CARDIAC = "cardiac"
    ORTHOPEDIC = "orthopedic"
    NEUROLOGICAL = "neurological"
    GENERAL = "general"
    PEDIATRIC = "pediatric"
    ONCOLOGICAL = "oncological"
    TRANSPLANT = "transplant"
    EMERGENCY = "emergency"
    MINIMALLY_INVASIVE = "minimally_invasive"


class PatientProfile(BaseModel):
    """Patient profile for surgical prediction."""
    age: int = Field(..., ge=0, le=120)
    gender: str = Field(..., pattern="^(male|female|other)$")
    bmi: float = Field(..., ge=10, le=60)
    medical_history: List[str] = []
    current_medications: List[str] = []
    allergies: List[str] = []
    lab_results: Dict[str, float] = {}
    vital_signs: Dict[str, float] = {}
    risk_factors: List[str] = []


class SurgicalCase(BaseModel):
    """Surgical case data for training."""
    case_id: str
    patient_profile: PatientProfile
    surgery_type: SurgeryType
    procedure_name: str
    urgency: str = Field(..., pattern="^(elective|urgent|emergency)$")
    
    # Surgical details
    surgeon_experience_years: int
    hospital_volume: str = Field(..., pattern="^(high|medium|low)$")
    equipment_available: List[str]
    technique_used: str
    
    # Outcomes
    duration_hours: float
    blood_loss_ml: float
    complications: List[str]
    success: bool
    recovery_days: int
    readmission_30_days: bool
    
    # Recommendations (for training)
    recommended_approach: str
    key_considerations: List[str]
    risk_mitigation: List[str]


class TrainingDataPoint(BaseModel):
    """Single training data point for FL."""
    instruction: str  # The medical query
    input: str  # Patient and surgical context
    response: str  # Expert recommendation


class SurgicalDataBatch(BaseModel):
    """Batch of surgical data for federated training."""
    hospital_id: str
    batch_id: str
    data_points: List[TrainingDataPoint]
    metadata: Dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.now)


def create_surgical_prompt(case: SurgicalCase) -> TrainingDataPoint:
    """Convert surgical case to training prompt."""
    
    # Construct instruction
    instruction = f"""Given a patient profile and surgical requirements, provide optimal surgical approach recommendations for a {case.surgery_type.value} {case.procedure_name} procedure."""
    
    # Construct input context
    input_context = f"""
Patient Profile:
- Age: {case.patient_profile.age} years
- Gender: {case.patient_profile.gender}
- BMI: {case.patient_profile.bmi}
- Medical History: {', '.join(case.patient_profile.medical_history) if case.patient_profile.medical_history else 'None'}
- Current Medications: {', '.join(case.patient_profile.current_medications) if case.patient_profile.current_medications else 'None'}
- Risk Factors: {', '.join(case.patient_profile.risk_factors) if case.patient_profile.risk_factors else 'None'}

Surgical Context:
- Procedure: {case.procedure_name}
- Type: {case.surgery_type.value}
- Urgency: {case.urgency}
- Surgeon Experience: {case.surgeon_experience_years} years
- Hospital Volume: {case.hospital_volume}
- Available Equipment: {', '.join(case.equipment_available)}

Key Lab Results:
{format_lab_results(case.patient_profile.lab_results)}
"""
    
    # Construct expert response
    response = f"""
Recommended Surgical Approach:
{case.recommended_approach}

Duration Estimate: {case.duration_hours:.1f} hours
Expected Blood Loss: {case.blood_loss_ml:.0f} ml
Expected Recovery: {case.recovery_days} days

Key Considerations:
{format_list_items(case.key_considerations)}

Risk Mitigation Strategies:
{format_list_items(case.risk_mitigation)}

Potential Complications to Monitor:
{format_list_items(case.complications)}

Success Factors:
- Technique: {case.technique_used}
- Equipment optimization for available resources
- Post-operative care plan focusing on {get_recovery_focus(case)}
"""
    
    return TrainingDataPoint(
        instruction=instruction.strip(),
        input=input_context.strip(),
        response=response.strip()
    )


def format_lab_results(lab_results: Dict[str, float]) -> str:
    """Format lab results for prompt."""
    if not lab_results:
        return "- No specific lab results provided"
    
    formatted = []
    for test, value in lab_results.items():
        formatted.append(f"- {test}: {value}")
    return "\n".join(formatted)


def format_list_items(items: List[str]) -> str:
    """Format list items for prompt."""
    if not items:
        return "- None identified"
    
    return "\n".join([f"- {item}" for item in items])


def get_recovery_focus(case: SurgicalCase) -> str:
    """Determine recovery focus based on surgery type."""
    recovery_focus = {
        SurgeryType.CARDIAC: "cardiac rehabilitation and monitoring",
        SurgeryType.ORTHOPEDIC: "physical therapy and mobility",
        SurgeryType.NEUROLOGICAL: "neurological monitoring and cognitive recovery",
        SurgeryType.GENERAL: "wound healing and infection prevention",
        SurgeryType.PEDIATRIC: "age-appropriate recovery and family support",
        SurgeryType.ONCOLOGICAL: "oncological follow-up and immune support",
        SurgeryType.TRANSPLANT: "immunosuppression management and rejection monitoring",
        SurgeryType.EMERGENCY: "stabilization and complication prevention",
        SurgeryType.MINIMALLY_INVASIVE: "early mobilization and rapid recovery"
    }
    return recovery_focus.get(case.surgery_type, "standard post-operative care")