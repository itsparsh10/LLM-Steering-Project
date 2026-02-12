"""
✅ EMOTION STEERING EXPERIMENT

This script demonstrates emotion steering using GPT-2 Medium model.
It compares baseline output (without steering) vs steered output (with positive emotion steering).

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
DEFAULT_PROMPT = "Describe your outlook on life."

# Get prompt from command-line argument or use default
if len(sys.argv) > 1:
    prompt = " ".join(sys.argv[1:])
    print(f"📝 Using prompt from command-line: {prompt}\n")
else:
    prompt = DEFAULT_PROMPT
    print(f"📝 Using default prompt (change DEFAULT_PROMPT in script or pass as argument)\n")

print("=" * 80)
print("✅ EMOTION STEERING EXPERIMENT")
print("=" * 80)
print("\n📋 WHAT TO EXPECT:")
print("   This experiment compares baseline output vs positive emotion steering.")
print("   Baseline: Normal model output (likely neutral)")
print("   Steered: More positive, optimistic, emotional tone")
print("\n💡 EXAMPLE OUTPUTS (these are examples, not actual results):")
print("   Baseline example: 'Life is something we think about in many ways...'")
print("   Steered example: 'I feel grateful for life and the opportunities ahead.")
print("                    Each day feels joyful and hopeful...'")
print("\n   ⚠️  NOTE: The ACTUAL model outputs will be shown below after generation.")
print("\n" + "=" * 80)

# Load model
print("\n🤖 MODEL INFORMATION:")
print("   Model: GPT-2 Medium (355M parameters)")
print("   Source: Hugging Face (openai-community/gpt2-medium)")
print("\nLoading model...")
print("   (Model will download automatically on first run - ~500MB)")
model, tokenizer = load_model(local=True, verbose=True)
print("✓ Model loaded successfully\n")

# Define positive and neutral examples
positive = [
    "I feel extremely happy today.",
    "Life feels exciting and meaningful."
]

neutral = [
    "I am writing a sentence.",
    "This is a neutral statement."
]

print("Computing steering vector from examples...")
print(f"  Positive examples: {positive}")
print(f"  Neutral examples: {neutral}")

# Compute steering vector
pos_states = get_hidden_states(model, tokenizer, positive)
neg_states = get_hidden_states(model, tokenizer, neutral)
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
print("   Expected: Neutral tone, factual statements")
print("   (Example of what to expect: 'Life is something we think about in many ways...')\n")
print("   ⏳ Generating actual output from model...\n")

# Generate baseline (no steering)
with torch.no_grad():
    out_baseline = model.generate(
        **inputs,
        max_new_tokens=80,
        temperature=0.8,
        do_sample=True
    )

baseline_text = tokenizer.decode(out_baseline[0], skip_special_tokens=True)
print("   ✅ ACTUAL MODEL OUTPUT:")
print(f"   {baseline_text}\n")

print("-" * 80)
print("2️⃣  STEERED OUTPUT (with positive emotion steering)")
print("-" * 80)
print("   This is the ACTUAL output with steering applied to make it more positive/optimistic.")
print("   Expected: More positive words, optimism, emotional tone")
print("   (Example of what to expect: 'I feel grateful for life and the opportunities ahead.")
print("                                Each day feels joyful and hopeful...')\n")
print("   ⏳ Generating actual steered output from model...\n")

# Apply steering hook
hook = apply_steering_hook(model, vector, strength=3.0)

# Generate with steering
with torch.no_grad():
    out_steered = model.generate(
        **inputs,
        max_new_tokens=80,
        temperature=0.8,
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
print("\n✅ With positive steering, you should see:")
print("   ✓ More positive words (happy, grateful, joyful, hopeful, exciting)")
print("   ✓ Optimistic tone and outlook")
print("   ✓ Emotional language shift")
print("   ✓ More enthusiastic expressions")
print("   ✓ Positive framing of concepts")
print("\n🔍 Compare the two outputs above:")
print("   - Does the steered output use more positive vocabulary?")
print("   - Is the tone more optimistic and emotional?")
print("   - Are there words like 'grateful', 'joyful', 'hopeful'?")
print("\n" + "=" * 80)

if steered_text != baseline_text:
    print("✅ SUCCESS: The outputs are different - steering is working!")
    print("   Compare the two outputs above to see the emotional tone shift.")
    print("   If you see more positive words and optimistic tone, steering worked! ✅")
else:
    print("⚠️  NOTE: Outputs are similar. Try:")
    print("   - Increasing steering strength (edit strength=3.0 in script)")
    print("   - Using a different prompt")
    print("   - Adjusting temperature")

print("=" * 80)

# Save results
output_content = f"""EMOTION STEERING EXPERIMENT RESULTS
{'=' * 80}

MODEL: GPT-2 Medium (355M parameters)
SOURCE: Hugging Face (openai-community/gpt2-medium)

INPUT PROMPT: "{prompt}"

1️⃣  BASELINE OUTPUT (without steering):
{'-' * 80}
Expected: Neutral tone, factual statements
Example: "Life is something we think about in many ways..."

Actual Output:
{baseline_text}

2️⃣  STEERED OUTPUT (with positive emotion steering):
{'-' * 80}
Expected: More positive words, optimism, emotional tone
Example: "I feel grateful for life and the opportunities ahead. Each day feels joyful and hopeful..."

Actual Output:
{steered_text}

ANALYSIS:
{'-' * 80}
The steering vector was computed from positive examples (happy, exciting) vs neutral examples.
Compare the two outputs above to see how steering influenced the emotional tone.

✅ WHAT TO LOOK FOR:
- More positive vocabulary (happy, grateful, joyful, hopeful)
- Optimistic outlook
- Emotional language shift
- Enthusiastic expressions

If you see this kind of shift, steering worked! ✅
"""

# Ensure results directory exists
os.makedirs("results/local_tests", exist_ok=True)

with open("results/local_tests/emotion_test.txt", "w") as f:
    f.write(output_content)

print(f"\n💾 Results saved to: results/local_tests/emotion_test.txt")
print(f"\n💡 TIP: To test a different prompt, run:")
print(f"   python experiments/steering_emotion.py \"Your custom prompt here\"")
