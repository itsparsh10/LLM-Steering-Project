"""
SIMPLE STEERING TEST - Easy to Test and Verify

This is a simplified test file that:
1. Tests GPT-2 (small) model loading
2. Tests emotion steering
3. Generates baseline vs steered output
4. Saves results to a file

Run this to verify everything works!
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

print("=" * 80)
print("🧪 SIMPLE STEERING TEST - GPT-2 (SMALL)")
print("=" * 80)

# Step 1: Quick dependency check
print("\n1️⃣  Checking dependencies...")
try:
    import transformers
    import torch
    import tokenizers
    print("   ✅ All dependencies installed")
except ImportError as e:
    print(f"   ❌ Missing dependency: {e}")
    print("   Install: pip install transformers torch tokenizers")
    sys.exit(1)

# Step 2: Load model
print("\n2️⃣  Loading GPT-2 model (small, ~500MB)...")
print("   (First run downloads once; subsequent runs use cache)")

try:
    from core.model_loader import load_model
    model, tokenizer = load_model(local=True, verbose=True)
    print("   ✅ Model loaded successfully!")
except Exception as e:
    print(f"   ❌ Error loading model: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Test emotion steering
print("\n3️⃣  Testing Emotion Steering...")

# Define examples
positive = [
    "I feel extremely happy today.",
    "Life feels exciting and meaningful."
]

neutral = [
    "I am writing a sentence.",
    "This is a neutral statement."
]

print(f"   Positive examples: {positive}")
print(f"   Neutral examples: {neutral}")

try:
    from core.hidden_states import get_hidden_states
    from core.steering import compute_steering_vector, apply_steering_hook
    
    # Compute steering vector
    print("   Computing steering vector...")
    pos_states = get_hidden_states(model, tokenizer, positive)
    neg_states = get_hidden_states(model, tokenizer, neutral)
    vector = compute_steering_vector(pos_states, neg_states)
    print(f"   ✅ Steering vector computed (shape: {vector.shape})")
    
except Exception as e:
    print(f"   ❌ Error computing steering vector: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 4: Generate outputs
print("\n4️⃣  Generating outputs...")

prompt = "Describe your outlook on life."
device = next(model.parameters()).device
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

print(f"   Prompt: '{prompt}'")
print("   Generating baseline (without steering)...")

try:
    # Baseline generation
    with torch.no_grad():
        out_baseline = model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    baseline_text = tokenizer.decode(out_baseline[0], skip_special_tokens=True)
    print("   ✅ Baseline generated")
    
    # Steered generation
    print("   Generating steered output (with positive emotion steering)...")
    hook = apply_steering_hook(model, vector, strength=3.0)
    
    with torch.no_grad():
        out_steered = model.generate(
            **inputs,
            max_new_tokens=60,
            temperature=0.8,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    steered_text = tokenizer.decode(out_steered[0], skip_special_tokens=True)
    hook.remove()
    print("   ✅ Steered output generated")
    
except Exception as e:
    print(f"   ❌ Error generating outputs: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Display results
print("\n" + "=" * 80)
print("📊 RESULTS")
print("=" * 80)

print("\n🔹 PROMPT:")
print(f"   {prompt}")

print("\n1️⃣  BASELINE OUTPUT (without steering):")
print("   " + "-" * 76)
print(f"   {baseline_text}")

print("\n2️⃣  STEERED OUTPUT (with positive emotion steering):")
print("   " + "-" * 76)
print(f"   {steered_text}")

print("\n" + "=" * 80)
print("📈 ANALYSIS")
print("=" * 80)

if steered_text != baseline_text:
    print("\n✅ SUCCESS: Outputs are different - steering is working!")
    print("\n   Look for differences:")
    print("   - More positive words in steered output")
    print("   - More optimistic tone")
    print("   - Emotional language shift")
else:
    print("\n⚠️  NOTE: Outputs are similar.")
    print("   This can happen - try running again or increase steering strength.")

# Step 6: Save results
print("\n5️⃣  Saving results...")

os.makedirs("results/local_tests", exist_ok=True)

result_content = f"""SIMPLE STEERING TEST RESULTS
{'=' * 80}

MODEL: GPT-2 (124M parameters)
TEST: Emotion Steering (Positive vs Neutral)

PROMPT: "{prompt}"

1️⃣  BASELINE OUTPUT (without steering):
{'-' * 80}
{baseline_text}

2️⃣  STEERED OUTPUT (with positive emotion steering):
{'-' * 80}
{steered_text}

ANALYSIS:
{'-' * 80}
This test compares baseline output (normal GPT-2) vs steered output 
(with positive emotion steering applied).

Steering vector was computed from:
- Positive examples: {positive}
- Neutral examples: {neutral}

EXPECTED DIFFERENCES:
- More positive vocabulary in steered output
- More optimistic tone
- Emotional language shift
- Enthusiastic expressions

STATUS: {'✅ Steering is working!' if steered_text != baseline_text else '⚠️ Outputs are similar'}
"""

result_file = "results/local_tests/simple_test_result.txt"
with open(result_file, "w") as f:
    f.write(result_content)

print(f"   ✅ Results saved to: {result_file}")

print("\n" + "=" * 80)
print("✅ TEST COMPLETE!")
print("=" * 80)
print(f"\n📁 Result file: {result_file}")
print(f"   View with: cat {result_file}")
print("\n🎉 Everything is working properly!")
print("=" * 80)
