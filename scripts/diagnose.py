"""
Diagnostic script to check system status.
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_environment():
    print("=" * 50)
    print("SYSTEM DIAGNOSTIC")
    print("=" * 50)
    
    checks = []
    
    # Check Python version
    import platform
    checks.append(("Python Version", platform.python_version()))
    
    # Check key packages
    packages = {
        "fastapi": None,
        "flwr": None,
        "torch": None,
        "transformers": None,
        "numpy": None,
        "sqlalchemy": None,
        "pydantic": None,
    }
    
    for pkg_name in packages:
        try:
            pkg = __import__(pkg_name)
            version = getattr(pkg, "__version__", "installed")
            checks.append((f"{pkg_name}", f"[OK] {version}"))
        except ImportError:
            checks.append((f"{pkg_name}", "[FAIL] Not installed"))
    
    # Check app imports
    try:
        from app.core.config import settings
        checks.append(("App Config", "[OK] Loaded"))
    except Exception as e:
        checks.append(("App Config", f"[FAIL] {str(e)[:50]}"))
    
    # Check database
    try:
        from app.db.base import SessionLocal
        from sqlalchemy import text
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        checks.append(("Database", "[OK] Connected"))
    except Exception as e:
        checks.append(("Database", f"[FAIL] {str(e)[:50]}"))
    
    # Check file structure
    important_files = [
        ("Main App", "app/main.py"),
        ("Config", "app/core/config.py"),
        ("FL Config", "app/fl/config/model_config.yaml"),
        ("Simple Client", "app/fl/client/client_app_simple.py"),
        ("Mock Data", "data/hospitals/hospital_1/surgical_cases.json"),
        (".env", ".env"),
    ]
    
    for name, path in important_files:
        if os.path.exists(path):
            checks.append((name, f"[OK] {path}"))
        else:
            checks.append((name, f"[FAIL] Missing: {path}"))
    
    # Check Docker
    import subprocess
    try:
        result = subprocess.run(["docker", "ps"], capture_output=True, text=True, timeout=5)
        if "surgical-fl-postgres" in result.stdout:
            checks.append(("PostgreSQL Docker", "[OK] Running"))
        else:
            checks.append(("PostgreSQL Docker", "[FAIL] Not running"))
    except:
        checks.append(("Docker", "[FAIL] Not available"))
    
    # Print results
    print("\nSystem Status:")
    print("-" * 50)
    
    for name, status in checks:
        print(f"  {name:<20} : {status}")
    
    # Count issues
    issues = [c for c in checks if "[FAIL]" in c[1]]
    
    print("\n" + "=" * 50)
    if not issues:
        print("[SUCCESS] All checks passed! System ready.")
    else:
        print(f"[WARNING] Found {len(issues)} issues:")
        for name, issue in issues:
            print(f"  - {name}: {issue}")
        
        print("\nSuggested fixes:")
        if any("Docker" in i[0] for i in issues):
            print("  1. Start Docker: docker-compose up -d")
        if any("Not installed" in i[1] for i in issues):
            print("  2. Install packages: pip install -r requirements.txt")
        if any("Mock Data" in i[0] for i in issues):
            print("  3. Generate data: python scripts/generate_mock_data.py")
        if any(".env" in i[0] for i in issues):
            print("  4. Create .env file from .env.example")


if __name__ == "__main__":
    check_environment()