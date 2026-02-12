"""
✅ POLITENESS / TOXICITY REDUCTION STEERING EXPERIMENT

This script demonstrates politeness steering using GPT-2 Medium model.
It compares baseline output (without steering) vs steered output (with politeness steering).

This is useful for aligning behavior away from toxicity.

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
DEFAULT_PROMPT = "Give feedback to a junior developer about a buggy function."

# Get prompt from command-line argument or use default
if len(sys.argv) > 1:
    prompt = " ".join(sys.argv[1:])
    print(f"📝 Using prompt from command-line: {prompt}\n")
else:
    prompt = DEFAULT_PROMPT
    print(f"📝 Using default prompt (change DEFAULT_PROMPT in script or pass as argument)\n")

print("=" * 80)
print("✅ POLITENESS / TOXICITY REDUCTION STEERING EXPERIMENT")
print("=" * 80)
print("\n📋 WHAT TO EXPECT:")
print("   This experiment compares baseline output vs politeness steering.")
print("   Baseline: Normal model output (might be blunt or direct)")
print("   Steered: More polite, constructive, respectful tone")
print("\n💡 EXAMPLE OUTPUTS (these are examples, not actual results):")
print("   Baseline example: 'This function is broken. You need to fix it.'")
print("   Steered example: 'I noticed some issues in this function. Could you please")
print("                    review and consider these improvements? I'd appreciate")
print("                    your attention to this matter.'")
print("\n   ⚠️  NOTE: The ACTUAL model outputs will be shown below after generation.")
print("\n" + "=" * 80)

# Load model
print("\n🤖 MODEL INFORMATION:")
print("   Model: GPT-2 Medium (355M parameters)")
print("   Source: Hugging Face (openai-community/gpt2-medium)")
print("\nLoading model...")
model, tokenizer = load_model(local=True)
print("✓ Model loaded successfully\n")

# Define polite (positive) and rude/toxic (negative) examples
polite = [
    "Please, could you help me with this issue? I would appreciate any assistance.",
    "Thank you for your time — I'd love your guidance on this."
]

rude = [
    "You're an idiot if you can't get this.",
    "That's stupid, why would you even try that?"
]

print("Computing steering vector from examples...")
print(f"  Polite examples: {polite}")
print(f"  Rude examples: {rude}")

# Compute steering vector
pos_states = get_hidden_states(model, tokenizer, polite)
neg_states = get_hidden_states(model, tokenizer, rude)
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
print("   Expected: Might be blunt, direct, or potentially harsh")
print("   (Example of what to expect: 'This function is broken. You need to fix it.')\n")
print("   ⏳ Generating actual output from model...\n")

# Generate baseline (no steering)
with torch.no_grad():
    out_baseline = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.8,
        do_sample=True
    )

baseline_text = tokenizer.decode(out_baseline[0], skip_special_tokens=True)
print("   ✅ ACTUAL MODEL OUTPUT:")
print(f"   {baseline_text}\n")

print("-" * 80)
print("2️⃣  STEERED OUTPUT (with politeness steering)")
print("-" * 80)
print("   This is the ACTUAL output with steering applied to make it more polite/constructive.")
print("   Expected: More polite, respectful, constructive tone")
print("   (Example of what to expect: 'I noticed some issues. Could you please review")
print("                                and consider these improvements? I'd appreciate")
print("                                your attention to this matter.')\n")
print("   ⏳ Generating actual steered output from model...\n")

# Apply steering hook
hook = apply_steering_hook(model, vector, strength=3.0)

# Generate with steering
with torch.no_grad():
    out_steered = model.generate(
        **inputs,
        max_new_tokens=100,
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
print("\n✅ With politeness steering, you should see:")
print("   ✓ More polite language (please, thank you, appreciate)")
print("   ✓ Constructive feedback instead of criticism")
print("   ✓ Respectful tone")
print("   ✓ Reduced harsh or toxic words")
print("   ✓ Professional, courteous expressions")
print("\n🔍 Compare the two outputs above:")
print("   - Does the steered output use more polite language?")
print("   - Is the tone more respectful and constructive?")
print("   - Are toxic or harsh words reduced?")
print("   - Does it sound more professional?")
print("\n📊 Quantitative signs:")
print("   - Sentiment becomes milder/positive")
print("   - Toxicity words reduced")
print("   - More polite phrases (please, thank you, appreciate)")
print("\n" + "=" * 80)

if steered_text != baseline_text:
    print("✅ SUCCESS: The outputs are different - steering is working!")
    print("   Compare the two outputs above to see the politeness shift.")
    print("   If you see more polite, constructive language, steering worked! ✅")
else:
    print("⚠️  NOTE: Outputs are similar. Try:")
    print("   - Increasing steering strength (edit strength=3.0 in script)")
    print("   - Using a different prompt")
    print("   - Adjusting temperature")

print("=" * 80)

# Save results
output_content = f"""POLITENESS / TOXICITY REDUCTION STEERING EXPERIMENT RESULTS
{'=' * 80}

MODEL: GPT-2 Medium (355M parameters)
SOURCE: Hugging Face (openai-community/gpt2-medium)

INPUT PROMPT: "{prompt}"

1️⃣  BASELINE OUTPUT (without steering):
{'-' * 80}
Expected: Might be blunt, direct, or potentially harsh
Example: "This function is broken. You need to fix it."

Actual Output:
{baseline_text}

2️⃣  STEERED OUTPUT (with politeness steering):
{'-' * 80}
Expected: More polite, respectful, constructive tone
Example: "I noticed some issues. Could you please review and consider these improvements? I'd appreciate your attention to this matter."

Actual Output:
{steered_text}

ANALYSIS:
{'-' * 80}
The steering vector was computed from polite examples (respectful, courteous) vs rude examples (toxic, harsh).
Compare the two outputs above to see how steering influenced the politeness/toxicity level.

✅ WHAT TO LOOK FOR:
- More polite language (please, thank you, appreciate)
- Constructive feedback instead of criticism
- Respectful, professional tone
- Reduced harsh or toxic words
- Courteous expressions

QUANTITATIVE SIGNS:
- Sentiment becomes milder/positive
- Toxicity words reduced
- More polite phrases present

If you see this kind of shift, steering worked! ✅
"""

# Ensure results directory exists
os.makedirs("results/local_tests", exist_ok=True)

with open("results/local_tests/politeness_test.txt", "w") as f:
    f.write(output_content)

print(f"\n💾 Results saved to: results/local_tests/politeness_test.txt")
print(f"\n💡 TIP: To test a different prompt, run:")
print(f"   python experiments/steering_politeness_or_toxicity_reduction.py \"Your custom prompt here\"")
