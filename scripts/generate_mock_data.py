"""
Generate mock surgical data for testing federated learning.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import random
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict

from app.fl.common.data_structures import SurgicalCase, PatientProfile, SurgeryType
from app.utils.logger import app_logger


def generate_patient_profile() -> Dict:
    """Generate a random patient profile."""
    age = random.randint(18, 85)
    gender = random.choice(["male", "female"])
    bmi = round(random.uniform(18.5, 35.0), 1)
    
    # Medical history based on age
    medical_conditions = ["Hypertension", "Diabetes", "Heart Disease", "Asthma", "Arthritis"]
    if age > 50:
        medical_history = random.sample(medical_conditions, k=random.randint(0, 3))
    else:
        medical_history = random.sample(medical_conditions, k=random.randint(0, 1))
    
    # Medications
    medications = ["Aspirin", "Metformin", "Lisinopril", "Atorvastatin", "Omeprazole"]
    current_medications = random.sample(medications, k=len(medical_history))
    
    # Lab results
    lab_results = {
        "Hemoglobin": round(random.uniform(12.0, 16.0), 1),
        "WBC": round(random.uniform(4.5, 11.0), 1),
        "Platelets": random.randint(150, 400),
        "Creatinine": round(random.uniform(0.6, 1.2), 2),
        "INR": round(random.uniform(0.9, 1.1), 2),
    }
    
    # Vital signs
    vital_signs = {
        "BP_systolic": random.randint(110, 140),
        "BP_diastolic": random.randint(70, 90),
        "Heart_rate": random.randint(60, 100),
        "Temperature": round(random.uniform(36.5, 37.5), 1),
        "SpO2": random.randint(95, 100),
    }
    
    # Risk factors
    risk_factors = []
    if bmi > 30:
        risk_factors.append("Obesity")
    if age > 65:
        risk_factors.append("Advanced Age")
    if random.random() < 0.3:
        risk_factors.append("Smoking")
    if "Diabetes" in medical_history:
        risk_factors.append("Diabetes")
    
    return {
        "age": age,
        "gender": gender,
        "bmi": bmi,
        "medical_history": medical_history,
        "current_medications": current_medications,
        "allergies": random.sample(["Penicillin", "Sulfa", "None"], k=1),
        "lab_results": lab_results,
        "vital_signs": vital_signs,
        "risk_factors": risk_factors,
    }


def generate_surgical_case(case_id: int, surgery_type: SurgeryType) -> SurgicalCase:
    """Generate a single surgical case."""
    
    # Procedure names by type
    procedures = {
        SurgeryType.CARDIAC: ["CABG", "Valve Replacement", "Angioplasty", "Pacemaker Implantation"],
        SurgeryType.ORTHOPEDIC: ["Total Knee Replacement", "Hip Replacement", "Spinal Fusion", "ACL Repair"],
        SurgeryType.NEUROLOGICAL: ["Craniotomy", "Spinal Decompression", "Tumor Resection", "VP Shunt"],
        SurgeryType.GENERAL: ["Cholecystectomy", "Appendectomy", "Hernia Repair", "Bowel Resection"],
        SurgeryType.ONCOLOGICAL: ["Mastectomy", "Lung Resection", "Colon Resection", "Liver Resection"],
    }
    
    procedure_name = random.choice(procedures.get(surgery_type, ["General Surgery"]))
    
    # Generate patient profile
    patient_profile = PatientProfile(**generate_patient_profile())
    
    # Surgery details
    urgency = random.choices(["elective", "urgent", "emergency"], weights=[0.7, 0.25, 0.05])[0]
    surgeon_experience = random.randint(3, 30)
    hospital_volume = random.choices(["high", "medium", "low"], weights=[0.3, 0.5, 0.2])[0]
    
    # Equipment
    basic_equipment = ["Standard OR Equipment", "Monitoring Systems", "Anesthesia Machine"]
    specialized_equipment = {
        SurgeryType.CARDIAC: ["Heart-Lung Machine", "ECMO", "IABP"],
        SurgeryType.ORTHOPEDIC: ["C-Arm", "Navigation System", "Power Tools"],
        SurgeryType.NEUROLOGICAL: ["Microscope", "Navigation System", "Neuromonitoring"],
        SurgeryType.GENERAL: ["Laparoscopic Equipment", "Energy Devices"],
    }
    
    equipment = basic_equipment + random.sample(
        specialized_equipment.get(surgery_type, []), 
        k=min(2, len(specialized_equipment.get(surgery_type, [])))
    )
    
    # Technique
    technique = random.choice(["Standard", "Minimally Invasive", "Robotic-Assisted"])
    
    # Outcomes (realistic based on procedure complexity)
    base_duration = {
        SurgeryType.CARDIAC: 4.0,
        SurgeryType.ORTHOPEDIC: 2.5,
        SurgeryType.NEUROLOGICAL: 5.0,
        SurgeryType.GENERAL: 1.5,
        SurgeryType.ONCOLOGICAL: 3.0,
    }
    
    duration = base_duration.get(surgery_type, 2.0) + random.uniform(-0.5, 1.5)
    
    # Blood loss (ml)
    base_blood_loss = {
        SurgeryType.CARDIAC: 400,
        SurgeryType.ORTHOPEDIC: 300,
        SurgeryType.NEUROLOGICAL: 200,
        SurgeryType.GENERAL: 100,
        SurgeryType.ONCOLOGICAL: 250,
    }
    
    blood_loss = base_blood_loss.get(surgery_type, 150) + random.randint(-50, 150)
    
    # Complications (lower rate for experienced surgeons)
    complication_rate = 0.1 if surgeon_experience > 10 else 0.2
    complications = []
    if random.random() < complication_rate:
        possible_complications = ["Minor bleeding", "Infection", "Delayed healing", "Pain"]
        complications = random.sample(possible_complications, k=random.randint(1, 2))
    
    # Success (higher for experienced surgeons and high-volume hospitals)
    success_rate = 0.95 if surgeon_experience > 10 and hospital_volume == "high" else 0.90
    success = random.random() < success_rate
    
    # Recovery
    base_recovery = {
        SurgeryType.CARDIAC: 7,
        SurgeryType.ORTHOPEDIC: 5,
        SurgeryType.NEUROLOGICAL: 10,
        SurgeryType.GENERAL: 3,
        SurgeryType.ONCOLOGICAL: 7,
    }
    
    recovery_days = base_recovery.get(surgery_type, 5) + random.randint(-2, 3)
    recovery_days = max(1, recovery_days)
    
    # Readmission
    readmission_rate = 0.05 if success and not complications else 0.15
    readmission = random.random() < readmission_rate
    
    # Recommendations
    approach_templates = {
        SurgeryType.CARDIAC: "Median sternotomy with cardiopulmonary bypass. Consider minimally invasive approach if suitable.",
        SurgeryType.ORTHOPEDIC: "Standard surgical approach with attention to alignment and soft tissue balance.",
        SurgeryType.NEUROLOGICAL: "Microsurgical technique with neuromonitoring. Preserve neural structures.",
        SurgeryType.GENERAL: "Laparoscopic approach preferred if feasible. Convert to open if necessary.",
        SurgeryType.ONCOLOGICAL: "Wide excision with negative margins. Consider sentinel node biopsy.",
    }
    
    considerations = [
        "Optimize patient's medical conditions preoperatively",
        "Ensure appropriate antibiotic prophylaxis",
        "Monitor for procedure-specific complications",
        "Early mobilization when appropriate",
    ]
    
    risk_mitigation = [
        "DVT prophylaxis per protocol",
        "Multimodal analgesia to minimize opioid use",
        "Strict sterile technique",
        "Clear communication with anesthesia team",
    ]
    
    return SurgicalCase(
        case_id=f"CASE_{case_id:06d}",
        patient_profile=patient_profile,
        surgery_type=surgery_type,
        procedure_name=procedure_name,
        urgency=urgency,
        surgeon_experience_years=surgeon_experience,
        hospital_volume=hospital_volume,
        equipment_available=equipment,
        technique_used=technique,
        duration_hours=round(duration, 1),
        blood_loss_ml=blood_loss,
        complications=complications if complications else ["None"],
        success=success,
        recovery_days=recovery_days,
        readmission_30_days=readmission,
        recommended_approach=approach_templates.get(surgery_type, "Standard surgical approach"),
        key_considerations=considerations,
        risk_mitigation=risk_mitigation,
    )


def generate_hospital_data(hospital_id: int, num_cases: int = 100):
    """Generate surgical cases for a hospital."""
    
    # Create hospital directory
    hospital_dir = Path(f"./data/hospitals/hospital_{hospital_id}")
    hospital_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate cases with realistic distribution
    surgery_type_weights = {
        SurgeryType.GENERAL: 0.35,
        SurgeryType.ORTHOPEDIC: 0.25,
        SurgeryType.CARDIAC: 0.15,
        SurgeryType.ONCOLOGICAL: 0.15,
        SurgeryType.NEUROLOGICAL: 0.10,
    }
    
    cases = []
    for i in range(num_cases):
        # Choose surgery type based on weights
        surgery_type = random.choices(
            list(surgery_type_weights.keys()),
            weights=list(surgery_type_weights.values())
        )[0]
        
        case = generate_surgical_case(i, surgery_type)
        cases.append(case.dict())
    
    # Save cases
    output_file = hospital_dir / "surgical_cases.json"
    with open(output_file, 'w') as f:
        json.dump(cases, f, indent=2, default=str)
    
    app_logger.info(f"Generated {num_cases} cases for hospital {hospital_id}")
    app_logger.info(f"Saved to {output_file}")
    
    # Generate summary statistics
    stats = {
        "hospital_id": hospital_id,
        "total_cases": num_cases,
        "surgery_types": {},
        "success_rate": sum(1 for c in cases if c["success"]) / num_cases,
        "avg_duration": sum(c["duration_hours"] for c in cases) / num_cases,
        "avg_recovery": sum(c["recovery_days"] for c in cases) / num_cases,
        "readmission_rate": sum(1 for c in cases if c["readmission_30_days"]) / num_cases,
    }
    
    for surgery_type in SurgeryType:
        type_cases = [c for c in cases if c["surgery_type"] == surgery_type.value]
        if type_cases:
            stats["surgery_types"][surgery_type.value] = len(type_cases)
    
    # Save statistics
    stats_file = hospital_dir / "statistics.json"
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    return stats


def main():
    """Generate mock data for multiple hospitals."""
    
    app_logger.info("=" * 50)
    app_logger.info("Generating Mock Surgical Data")
    app_logger.info("=" * 50)
    
    # Number of hospitals and cases
    num_hospitals = 4
    cases_per_hospital = 100
    
    all_stats = []
    
    for hospital_id in range(1, num_hospitals + 1):
        app_logger.info(f"\nGenerating data for Hospital {hospital_id}...")
        stats = generate_hospital_data(hospital_id, cases_per_hospital)
        all_stats.append(stats)
    
    # Display summary
    app_logger.info("\n" + "=" * 50)
    app_logger.info("Generation Summary")
    app_logger.info("=" * 50)
    
    for stats in all_stats:
        app_logger.info(f"\nHospital {stats['hospital_id']}:")
        app_logger.info(f"  Total cases: {stats['total_cases']}")
        app_logger.info(f"  Success rate: {stats['success_rate']:.2%}")
        app_logger.info(f"  Avg duration: {stats['avg_duration']:.1f} hours")
        app_logger.info(f"  Avg recovery: {stats['avg_recovery']:.1f} days")
        app_logger.info(f"  Readmission rate: {stats['readmission_rate']:.2%}")
        app_logger.info(f"  Case distribution: {stats['surgery_types']}")
    
    app_logger.info("\n✓ Mock data generation completed successfully!")
    app_logger.info(f"Data saved in: ./data/hospitals/")


if __name__ == "__main__":
    main()