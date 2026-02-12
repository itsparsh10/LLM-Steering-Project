"""
Auto-setup module for experiments.
Checks if model is ready and sets up if needed.
OPTIMIZED FOR SPEED - Only checks dependencies, model loads in experiment.
"""

import sys
import subprocess
import os

def check_dependencies():
    """Check if required dependencies are installed."""
    required = ['transformers', 'torch', 'tokenizers']
    missing = []
    
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    return missing

def install_dependencies(missing_packages):
    """Install missing dependencies."""
    print(f"\n📥 Installing missing packages: {', '.join(missing_packages)}")
    print("   (This may take a few minutes...)")
    
    for package in missing_packages:
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                package, "--user", "--quiet"
            ])
            print(f"   ✅ {package} installed")
        except subprocess.CalledProcessError:
            print(f"   ❌ Failed to install {package}")
            return False
    return True

def ensure_setup():
    """Ensure dependencies are set up - FAST VERSION.
    
    Only checks dependencies. Model will be loaded by the experiment script
    itself to avoid double loading and save time.
    """
    # Quick dependency check only
    missing = check_dependencies()
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("   Installing...")
        if not install_dependencies(missing):
            print("\n❌ Failed to install dependencies!")
            print("   Please run: python3 ai_model.py")
            sys.exit(1)
        print("✅ Dependencies installed!")
    
    # Model will be loaded by the experiment script itself (avoids double loading)
    return True
