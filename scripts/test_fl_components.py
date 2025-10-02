"""
Test federated learning components individually.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
from app.utils.logger import app_logger
from app.fl.client.dataset import SurgicalDatasetLoader, load_hospital_data
from app.fl.common.data_structures import create_surgical_prompt
from app.services.training import fl_orchestration_service
from app.db.base import SessionLocal
from app.models.hospital import Hospital
from app.models.user import User, UserRole
import json


def test_data_loading():
    """Test loading surgical data."""
    app_logger.info("Testing data loading...")
    
    try:
        # Test for hospital 1
        loader = SurgicalDatasetLoader(hospital_id="1")
        cases = loader.load_surgical_cases()
        
        if cases:
            app_logger.info(f"✓ Loaded {len(cases)} surgical cases")
            
            # Test prompt creation
            training_data = loader.prepare_training_data(cases[:5])
            app_logger.info(f"✓ Created {len(training_data)} training examples")
            
            # Display sample
            if training_data:
                sample = training_data[0]
                app_logger.info("\nSample training data:")
                app_logger.info(f"Instruction: {sample.instruction[:100]}...")
                app_logger.info(f"Input length: {len(sample.input)} chars")
                app_logger.info(f"Response length: {len(sample.response)} chars")
            
            return True
        else:
            app_logger.error("No cases loaded")
            return False
            
    except Exception as e:
        app_logger.error(f"Data loading failed: {e}")
        return False


def test_model_loading():
    """Test model loading (lightweight test)."""
    app_logger.info("Testing model configuration...")
    
    try:
        from omegaconf import OmegaConf
        
        # Load config
        config_path = "app/fl/config/model_config.yaml"
        if os.path.exists(config_path):
            cfg = OmegaConf.load(config_path)
            app_logger.info(f"✓ Loaded configuration for model: {cfg.model.name}")
            app_logger.info(f"  Quantization: {cfg.model.quantization}-bit")
            app_logger.info(f"  LoRA rank: {cfg.model.lora.peft_lora_r}")
            return True
        else:
            app_logger.error(f"Config file not found: {config_path}")
            return False
            
    except Exception as e:
        app_logger.error(f"Model configuration test failed: {e}")
        return False


def test_database_hospitals():
    """Test hospital records in database."""
    app_logger.info("Testing hospital database records...")
    
    try:
        with SessionLocal() as db:
            # Check if hospitals exist
            hospitals = db.query(Hospital).all()
            
            if not hospitals:
                app_logger.info("Creating test hospitals...")
                
                # Create test hospitals
                for i in range(1, 5):
                    hospital = Hospital(
                        code=f"HOSP{i:03d}",
                        name=f"Test Hospital {i}",
                        city=f"City {i}",
                        state="Test State",
                        country="Test Country",
                        type="General",
                        bed_capacity=100 + i * 50,
                        annual_surgeries=1000 + i * 100,
                        fl_client_id=f"hospital_{i}",
                        is_active=True,
                        is_approved=True,
                        specialties=["General", "Cardiac", "Orthopedic"]
                    )
                    db.add(hospital)
                
                db.commit()
                hospitals = db.query(Hospital).all()
                app_logger.info(f"✓ Created {len(hospitals)} test hospitals")
            else:
                app_logger.info(f"✓ Found {len(hospitals)} hospitals in database")
            
            # Display hospitals
            for h in hospitals:
                app_logger.info(f"  - {h.code}: {h.name} (Active: {h.is_active}, Approved: {h.is_approved})")
            
            return True
            
    except Exception as e:
        app_logger.error(f"Hospital database test failed: {e}")
        return False


async def test_orchestration_service():
    """Test FL orchestration service."""
    app_logger.info("Testing FL orchestration service...")
    
    try:
        # List active sessions
        activonfe_sessions = fl_orchestration_service.list_active_sessions()
        app_logger.info(f"Active sessions: {len(active_sessions)}")
        
        # Test starting a training session (mock)
        app_logger.info("✓ Orchestration service is functional")
        
        return True
        
    except Exception as e:
        app_logger.error(f"Orchestration service test failed: {e}")
        return False


def main():
    """Run all FL component tests."""
    app_logger.info("=" * 50)
    app_logger.info("Testing FL Components")
    app_logger.info("=" * 50)
    
    tests = [
        ("Data Loading", test_data_loading),
        ("Model Configuration", test_model_loading),
        ("Hospital Database", test_database_hospitals),
        ("Orchestration Service", lambda: asyncio.run(test_orchestration_service())),
    ]
    
    results = []
    for test_name, test_func in tests:
        app_logger.info(f"\n{test_name}:")
        app_logger.info("-" * 30)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            app_logger.error(f"Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    app_logger.info("\n" + "=" * 50)
    app_logger.info("Test Summary")
    app_logger.info("=" * 50)
    
    all_passed = True
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        app_logger.info(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        app_logger.info("\n🎉 All FL component tests passed!")
        app_logger.info("\nYou can now:")
        app_logger.info("1. Start the FL server manually for testing")
        app_logger.info("2. Run FL clients for each hospital")
        app_logger.info("3. Or use the orchestration service via API")
    else:
        app_logger.error("\n❌ Some tests failed. Please check the errors above.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)