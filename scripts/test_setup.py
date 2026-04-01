#!/usr/bin/env python3
"""
Simple test to verify project structure and imports.
Run this before attempting full training.
"""

import sys
from pathlib import Path

def check_project_structure():
    """Check if all required directories and files exist."""
    print("🔍 Checking project structure...")

    required_paths = [
        "config/model_config.yaml",
        "models/architecture/skawr_transformer.py",
        "models/training/trainer.py",
        "models/training/dataset.py",
        "scripts/train_model.py",
        "scripts/data_collection/kaggle_downloader.py",
        "scripts/data_collection/reddit_scraper.py",
        "requirements.txt"
    ]

    missing_paths = []
    for path in required_paths:
        if not Path(path).exists():
            missing_paths.append(path)

    if missing_paths:
        print("❌ Missing required files:")
        for path in missing_paths:
            print(f"  - {path}")
        return False
    else:
        print("✅ All required files present")
        return True

def check_config():
    """Check if config file is valid."""
    print("\n📋 Checking configuration...")

    try:
        import yaml
        with open("config/model_config.yaml", 'r') as f:
            config = yaml.safe_load(f)

        required_sections = ["model", "training", "data"]
        for section in required_sections:
            if section not in config:
                print(f"❌ Missing config section: {section}")
                return False

        print("✅ Configuration file valid")
        return True

    except Exception as e:
        print(f"❌ Config error: {e}")
        return False

def test_imports():
    """Test if core Python imports work."""
    print("\n🐍 Testing core imports...")

    try:
        import pandas as pd
        print("✅ pandas available")
    except ImportError:
        print("❌ pandas not installed")
        return False

    try:
        import numpy as np
        print("✅ numpy available")
    except ImportError:
        print("❌ numpy not installed")
        return False

    try:
        import yaml
        print("✅ yaml available")
    except ImportError:
        print("❌ yaml not installed")
        return False

    return True

def test_ml_imports():
    """Test ML-specific imports (optional)."""
    print("\n🧠 Testing ML imports (optional)...")

    ml_status = {
        "torch": False,
        "transformers": False,
        "sklearn": False
    }

    try:
        import torch
        ml_status["torch"] = True
        print(f"✅ PyTorch {torch.__version__} available")
    except ImportError:
        print("⚠️  PyTorch not installed (required for training)")

    try:
        import transformers
        ml_status["transformers"] = True
        print(f"✅ Transformers {transformers.__version__} available")
    except ImportError:
        print("⚠️  Transformers not installed (required for training)")

    try:
        import sklearn
        ml_status["sklearn"] = True
        print("✅ scikit-learn available")
    except ImportError:
        print("⚠️  scikit-learn not installed")

    return ml_status

def create_dummy_data_test():
    """Test creating dummy data without ML dependencies."""
    print("\n📊 Testing dummy data creation...")

    try:
        import pandas as pd
        import random

        # Create minimal dummy data
        data = []
        domains = ["automotive", "electronics"]

        for i in range(10):
            domain = random.choice(domains)
            data.append({
                'text': f'Sample {domain} product {i}',
                'domain': domain,
                'category': f'{domain}_category',
                'source': 'test_data'
            })

        df = pd.DataFrame(data)

        # Save to test directory
        test_dir = Path("test_data")
        test_dir.mkdir(exist_ok=True)

        df.to_csv(test_dir / "test_train.csv", index=False)
        print(f"✅ Created test data: {len(df)} samples")

        return True

    except Exception as e:
        print(f"❌ Dummy data creation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("🚀 SKAWR ML Setup Test\n")

    tests = [
        ("Project Structure", check_project_structure),
        ("Configuration", check_config),
        ("Core Imports", test_imports),
        ("Dummy Data", create_dummy_data_test)
    ]

    results = {}

    for test_name, test_func in tests:
        results[test_name] = test_func()

    # ML imports (optional)
    ml_status = test_ml_imports()
    results.update(ml_status)

    # Summary
    print(f"\n📋 Test Summary:")
    print(f"{'='*50}")

    required_tests = ["Project Structure", "Configuration", "Core Imports", "Dummy Data"]
    optional_tests = ["torch", "transformers", "sklearn"]

    all_required_passed = all(results[test] for test in required_tests)
    some_ml_available = any(results[test] for test in optional_tests)

    print("Required components:")
    for test in required_tests:
        status = "✅ PASS" if results[test] else "❌ FAIL"
        print(f"  {test}: {status}")

    print("\nOptional ML components:")
    for test in optional_tests:
        status = "✅ Available" if results[test] else "⚠️  Not installed"
        print(f"  {test}: {status}")

    print(f"\n{'='*50}")

    if all_required_passed:
        print("✅ SETUP READY: Core infrastructure is working")

        if some_ml_available:
            print("✅ ML READY: Can run training")
            print("\nNext steps:")
            print("  1. Install remaining ML dependencies: pip install torch transformers")
            print("  2. Run data collection: python scripts/data_collection/run_data_collection.py --demo-mode")
            print("  3. Test training: python scripts/train_model.py --create-dummy-data")
        else:
            print("⚠️  ML NOT READY: Install ML dependencies first")
            print("\nInstall command:")
            print("  pip install torch transformers scikit-learn wandb")
    else:
        print("❌ SETUP FAILED: Fix missing components first")

    print(f"\n{'='*50}")

if __name__ == "__main__":
    main()