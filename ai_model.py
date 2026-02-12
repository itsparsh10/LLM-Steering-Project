"""
AI Model Setup and Installation Script

This script handles:
1. Checking if dependencies are installed
2. Installing missing dependencies
3. Downloading and setting up GPT-2 Medium model
4. Verifying everything works

Run this once to set up everything, then you can run any experiment file directly.
"""

import sys
import subprocess
import os

def check_and_install_dependencies():
    """Check and install required dependencies."""
    print("=" * 80)
    print("📦 CHECKING DEPENDENCIES")
    print("=" * 80)
    
    required_packages = {
        'transformers': 'transformers',
        'torch': 'torch',
        'tokenizers': 'tokenizers',
        'numpy': 'numpy'
    }
    
    missing_packages = []
    
    for package_name, import_name in required_packages.items():
        try:
            __import__(import_name)
            print(f"   ✅ {package_name} installed")
        except ImportError:
            print(f"   ⚠️  {package_name} missing")
            missing_packages.append(package_name)
    
    if missing_packages:
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
    else:
        print("\n✅ All dependencies are installed!")
    
    return True

def setup_model():
    """Download and verify GPT-2 Medium model."""
    print("\n" + "=" * 80)
    print("🤖 SETTING UP GPT-2 MEDIUM MODEL")
    print("=" * 80)
    
    try:
        print("\n📥 Loading model (will download on first run)...")
        print("   Model: GPT-2 Medium (355M parameters)")
        print("   Source: Hugging Face")
        print("   Size: ~500MB (downloads automatically)")
        
        from core.model_loader import load_model
        
        # Load model with verbose output
        model, tokenizer = load_model(local=True, verbose=True)
        
        print("\n✅ Model setup complete!")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")
        print(f"   Device: {next(model.parameters()).device}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error setting up model: {e}")
        import traceback
        traceback.print_exc()
        return False

def verify_setup():
    """Verify everything is working."""
    print("\n" + "=" * 80)
    print("🧪 VERIFYING SETUP")
    print("=" * 80)
    
    try:
        from core.model_loader import load_model
        import torch
        
        print("\n1️⃣  Testing model loading...")
        model, tokenizer = load_model(local=True, verbose=False)
        print("   ✅ Model loads successfully")
        
        print("\n2️⃣  Testing tokenizer...")
        test_text = "Hello"
        tokens = tokenizer(test_text, return_tensors="pt")
        print(f"   ✅ Tokenizer works")
        
        print("\n3️⃣  Testing model forward pass...")
        device = next(model.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}
        
        with torch.no_grad():
            outputs = model(**tokens, output_hidden_states=True)
        print(f"   ✅ Forward pass works")
        print(f"   ✅ Hidden states available: {len(outputs.hidden_states)} layers")
        
        print("\n4️⃣  Testing text generation...")
        with torch.no_grad():
            generated = model.generate(
                **tokens,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        print("   ✅ Text generation works")
        
        print("\n" + "=" * 80)
        print("✅ SETUP COMPLETE - EVERYTHING IS READY!")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Verification failed: {e}")
        return False

def main():
    """Main setup function."""
    print("\n" + "=" * 80)
    print("🚀 AI MODEL SETUP - GPT-2 MEDIUM")
    print("=" * 80)
    print("\nThis script will:")
    print("  1. Check and install dependencies")
    print("  2. Download GPT-2 Medium model")
    print("  3. Verify everything works")
    print("\n" + "=" * 80)
    
    # Step 1: Check dependencies
    if not check_and_install_dependencies():
        print("\n❌ Dependency installation failed!")
        sys.exit(1)
    
    # Step 2: Setup model
    if not setup_model():
        print("\n❌ Model setup failed!")
        sys.exit(1)
    
    # Step 3: Verify
    if not verify_setup():
        print("\n❌ Verification failed!")
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("🎉 SETUP COMPLETE!")
    print("=" * 80)
    print("\n✅ You can now run any experiment file:")
    print("\n   python3 experiments/steering_emotion.py")
    print("   python3 experiments/steering_persona.py")
    print("   python3 experiments/steering_politeness_or_toxicity_reduction.py")
    print("   python3 experiments/steering_formality_conciseness.py")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
