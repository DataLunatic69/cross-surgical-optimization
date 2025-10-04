"""
Test federated learning components without TRL dependency.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import asyncio
import json
from pathlib import Path
from app.utils.logger import app_logger
from app.db.base import SessionLocal
from app.models.hospital import Hospital


def test_mock_data_exists():
    """Check if mock data files exist."""
    app_logger.info("Testing mock data existence...")
    
    try:
        data_exists = False
        for i in range(1, 5):
            data_path = Path(f"./data/hospitals/hospital_{i}/surgical_cases.json")
            if data_path.exists():
                with open(data_path, 'r') as f:
                    cases = json.load(f)
                app_logger.info(f"✓ Hospital {i}: {len(cases)} cases found")
                data_exists = True
            else:
                app_logger.warning(f"✗ Hospital {i}: No data found")
        
        if not data_exists:
            app_logger.info("\nPlease run: python scripts/generate_mock_data.py")
            return False
        
        return True
            
    except Exception as e:
        app_logger.error(f"Data check failed: {e}")
        return False


def test_config_exists():
    """Check if configuration files exist."""
    app_logger.info("Testing configuration files...")
    
    config_path = Path("app/fl/config/model_config.yaml")
    if config_path.exists():
        app_logger.info(f"✓ Config file exists: {config_path}")
        
        # Try to load it
        try:
            with open(config_path, 'r') as f:
                content = f.read()
                if content:
                    app_logger.info("✓ Config file is readable")
                    return True
        except Exception as e:
            app_logger.error(f"✗ Cannot read config: {e}")
            return False
    else:
        app_logger.error(f"✗ Config file not found: {config_path}")
        return False


def test_database_connection():
    """Test database connectivity."""
    app_logger.info("Testing database connection...")
    
    try:
        db = SessionLocal()
        from sqlalchemy import text
        db.execute(text("SELECT 1"))
        db.close()
        app_logger.info("✓ Database connection successful")
        return True
    except Exception as e:
        app_logger.error(f"✗ Database connection failed: {e}")
        app_logger.info("  Make sure PostgreSQL container is running")
        return False


def test_hospitals_in_db():
    """Check hospitals in database."""
    app_logger.info("Testing hospital records...")
    
    try:
        with SessionLocal() as db:
            hospitals = db.query(Hospital).all()
            
            if not hospitals:
                app_logger.info("No hospitals found. Creating test hospitals...")
                
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
                app_logger.info(f"✓ Created {len(hospitals)} hospitals")
            else:
                app_logger.info(f"✓ Found {len(hospitals)} hospitals")
            
            for h in hospitals:
                app_logger.info(f"  - {h.code}: {h.name}")
            
            return True
            
    except Exception as e:
        app_logger.error(f"✗ Hospital check failed: {e}")
        return False


def test_simple_fl_client():
    """Test the simplified FL client without TRL."""
    app_logger.info("Testing simplified FL client...")
    
    try:
        # Test direct import first
        import sys
        import importlib
        
        # Try to import the module
        import app.fl.client.client_app_simple as client_module
        
        # Test creating the client
        from app.fl.client.client_app_simple import SimpleSurgicalFLClient
        import numpy as np
        
        # Create a test client
        client = SimpleSurgicalFLClient(hospital_id="1", num_rounds=5)
        
        # Create mock parameters
        mock_params = [np.random.randn(100, 10).astype(np.float32) for _ in range(10)]
        
        # Test fit method
        config = {"current_round": 1}
        updated_params, num_samples, metrics = client.fit(mock_params, config)
        
        app_logger.info("✓ Simplified FL client works")
        app_logger.info(f"  Samples: {num_samples}")
        app_logger.info(f"  Loss: {metrics.get('train_loss', 0):.4f}")
        
        return True
        
    except ImportError as e:
        app_logger.error(f"✗ Cannot import simplified client: {e}")
        app_logger.info("  Creating the fixed client file...")
        
        # Try to create the file if it doesn't exist
        try:
            from pathlib import Path
            client_path = Path("app/fl/client/client_app_simple_fixed.py")
            if not client_path.exists():
                app_logger.info("  File will be created by fix script")
        except:
            pass
        
        return False
    except Exception as e:
        app_logger.error(f"✗ Client test failed: {e}")
        return False


def test_training_service():
    """Test if training service can be imported."""
    app_logger.info("Testing training service...")
    
    try:
        from app.services.training import fl_orchestration_service
        app_logger.info("✓ Training service can be imported")
        
        # Try to list sessions
        sessions = fl_orchestration_service.list_active_sessions()
        app_logger.info(f"✓ Active sessions: {len(sessions)}")
        
        return True
        
    except ImportError as e:
        app_logger.error(f"✗ Cannot import training service: {e}")
        return False
    except Exception as e:
        app_logger.error(f"✗ Training service test failed: {e}")
        return False


def main():
    """Run all tests."""
    app_logger.info("=" * 50)
    app_logger.info("FL Components Test (Simplified - No TRL)")
    app_logger.info("=" * 50)
    
    tests = [
        ("Mock Data Files", test_mock_data_exists),
        ("Configuration Files", test_config_exists),
        ("Database Connection", test_database_connection),
        ("Hospital Records", test_hospitals_in_db),
        ("Simplified FL Client", test_simple_fl_client),
        ("Training Service", test_training_service),
    ]
    
    results = []
    for test_name, test_func in tests:
        app_logger.info(f"\n{test_name}:")
        app_logger.info("-" * 30)
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            app_logger.error(f"Unexpected error: {e}")
            results.append((test_name, False))
    
    # Summary
    app_logger.info("\n" + "=" * 50)
    app_logger.info("Test Summary")
    app_logger.info("=" * 50)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        app_logger.info(f"{test_name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        app_logger.info("\n✅ All tests passed!")
        app_logger.info("\nNext: Start the API server")
        app_logger.info("  python -m uvicorn app.main:app --reload")
    else:
        app_logger.info("\n⚠️ Some tests failed")
        app_logger.info("\nSuggested fixes:")
        
        if not results[0][1]:  # Mock data failed
            app_logger.info("1. Generate mock data: python scripts/generate_mock_data.py")
        
        if not results[2][1]:  # Database failed
            app_logger.info("2. Start Docker: docker-compose up -d")
        
        if not results[4][1]:  # Client failed
            app_logger.info("3. Ensure client_app_simple.py was created from earlier steps")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)