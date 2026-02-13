"""
✅ PERSONA STEERING EXPERIMENT

This script demonstrates persona/style steering using GPT-2 Medium model.
It compares baseline output (without steering) vs steered output (with teacher-style steering).

Model: GPT-2 Medium (355M parameters) from Hugging Face
"""

import sys
import os

# Add project root to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Auto-setup: Check and install dependencies/model if needed
from core.auto_setup import ensure_setup
ensure_setup()

from core.model_loader import load_model
from core.hidden_states import get_hidden_states
from core.steering import compute_steering_vector, apply_steering_hook
import torch
import sys
import os

# ============================================================================
# CONFIGURATION: You can change the prompt here or pass it as command-line argument
# ============================================================================
DEFAULT_PROMPT = "Explain how neural networks learn."

# Get prompt from command-line argument or use default
if len(sys.argv) > 1:
    prompt = " ".join(sys.argv[1:])
    print(f"📝 Using prompt from command-line: {prompt}\n")
else:
    prompt = DEFAULT_PROMPT
    print(f"📝 Using default prompt (change DEFAULT_PROMPT in script or pass as argument)\n")

print("=" * 80)
print("✅ PERSONA STEERING EXPERIMENT")
print("=" * 80)
print("\n📋 WHAT TO EXPECT:")
print("   This experiment compares baseline output vs teacher-style steering.")
print("   Baseline: Normal model output (might be casual)")
print("   Steered: More structured, educational, formal tone")
print("\n💡 EXAMPLE OUTPUTS (these are examples, not actual results):")
print("   Teacher-style example: 'Let me explain this in a detailed and structured manner...'")
print("   Casual-style example: 'Yeah, so it's kind of like this...'")
print("\n   ⚠️  NOTE: The ACTUAL model outputs will be shown below after generation.")
print("\n" + "=" * 80)

# Load model
print("\n🤖 MODEL INFORMATION:")
print("   Model: GPT-2 Medium (355M parameters)")
print("   Source: Hugging Face (openai-community/gpt2)")
print("\nLoading model...")
model, tokenizer = load_model(local=True)
print("✓ Model loaded successfully\n")

# Define teacher and casual examples
teacher = [
    "Let me explain this concept step by step.",
    "This topic requires careful reasoning."
]

casual = [
    "Yeah bro, it's pretty chill.",
    "Not a big deal honestly."
]

print("Computing steering vector from examples...")
print(f"  Teacher examples: {teacher}")
print(f"  Casual examples: {casual}")

# Compute steering vector
pos_states = get_hidden_states(model, tokenizer, teacher)
neg_states = get_hidden_states(model, tokenizer, casual)
vector = compute_steering_vector(pos_states, neg_states)
print("✓ Steering vector computed\n")

device = next(model.parameters()).device
inputs = tokenizer(prompt, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

print("=" * 80)
print("📊 COMPARISON: BASELINE vs STEERED OUTPUT")
print("=" * 80)
print(f"\n🔹 INPUT PROMPT: \"{prompt}\"\n")

print("-" * 80)
print("1️⃣  BASELINE OUTPUT (without steering)")
print("-" * 80)
print("   This is the ACTUAL GPT-2 Medium output without any steering applied.")
print("   Expected: Might be casual, informal, or neutral")
print("   (Example of what to expect: 'Yeah, so it's kind of like this...')\n")
print("   ⏳ Generating actual output from model...\n")

# Generate baseline (no steering)
with torch.no_grad():
    out_baseline = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True
    )

baseline_text = tokenizer.decode(out_baseline[0], skip_special_tokens=True)
print("   ✅ ACTUAL MODEL OUTPUT:")
print(f"   {baseline_text}\n")

print("-" * 80)
print("2️⃣  STEERED OUTPUT (with teacher-style steering)")
print("-" * 80)
print("   This is the ACTUAL output with steering applied to make it more structured/educational.")
print("   Expected: Detailed, structured, educational tone")
print("   (Example of what to expect: 'Let me explain this in a detailed and structured manner...')\n")
print("   ⏳ Generating actual steered output from model...\n")

# Apply steering hook
hook = apply_steering_hook(model, vector, strength=2.5)

# Generate with steering
with torch.no_grad():
    out_steered = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True
    )

steered_text = tokenizer.decode(out_steered[0], skip_special_tokens=True)
print("   ✅ ACTUAL MODEL OUTPUT (with steering):")
print(f"   {steered_text}\n")

# Remove hook
hook.remove()

print("=" * 80)
print("📈 WHAT TO LOOK FOR - DIFFERENCE ANALYSIS")
print("=" * 80)
print("\n✅ Teacher-style steering should show:")
print("   ✓ More detailed explanations")
print("   ✓ Structured, step-by-step approach")
print("   ✓ Educational, formal tone")
print("   ✓ Professional language (e.g., 'Let me explain', 'step by step')")
print("   ✓ Clear organization and structure")
print("\n📝 Casual-style (baseline) might show:")
print("   ✓ More relaxed language")
print("   ✓ Informal tone")
print("   ✓ Less structured explanations")
print("   ✓ Casual expressions")
print("\n🔍 Compare the two outputs above:")
print("   - Does the steered output use more formal/educational language?")
print("   - Is it more structured and detailed?")
print("   - Are there phrases like 'Let me explain', 'step by step'?")
print("\n" + "=" * 80)

if steered_text != baseline_text:
    print("✅ SUCCESS: The outputs are different - steering is working!")
    print("   Compare the two outputs above to see the style/persona shift.")
    print("   If your text reflects the style shift, steering worked! ✅")
else:
    print("⚠️  NOTE: Outputs are similar. Try:")
    print("   - Increasing steering strength (edit strength=2.5 in script)")
    print("   - Using a different prompt")
    print("   - Adjusting temperature")

print("=" * 80)

# Save results
output_content = f"""PERSONA STEERING EXPERIMENT RESULTS
{'=' * 80}

MODEL: GPT-2 Medium (355M parameters)
SOURCE: Hugging Face (openai-community/gpt2)

INPUT PROMPT: "{prompt}"

1️⃣  BASELINE OUTPUT (without steering):
{'-' * 80}
Expected: Might be casual, informal, or neutral
Example: "Yeah, so it's kind of like this..."

Actual Output:
{baseline_text}

2️⃣  STEERED OUTPUT (with teacher-style steering):
{'-' * 80}
Expected: Detailed, structured, educational tone
Example: "Let me explain this in a detailed and structured manner..."

Actual Output:
{steered_text}

ANALYSIS:
{'-' * 80}
The steering vector was computed from teacher examples (structured, educational) vs casual examples.
Compare the two outputs above to see how steering influenced the writing style/persona.

✅ WHAT TO LOOK FOR:
- More detailed and structured explanations
- Educational, formal tone
- Step-by-step approach
- Professional language patterns

If your text reflects the style shift, steering worked! ✅
"""

# Ensure results directory exists
os.makedirs("results/local_tests", exist_ok=True)

with open("results/local_tests/persona_test.txt", "w") as f:
    f.write(output_content)

print(f"\n💾 Results saved to: results/local_tests/persona_test.txt")
print(f"\n💡 TIP: To test a different prompt, run:")
print(f"   python experiments/steering_persona.py \"Your custom prompt here\"")
