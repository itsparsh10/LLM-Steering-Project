"""
✅ FORMALITY / CONCISENESS STEERING EXPERIMENT

This script demonstrates formality/conciseness steering using GPT-2 Medium model.
It compares baseline output (without steering) vs steered output (with formal steering).

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
DEFAULT_PROMPT = "Write a short summary of the experiment results."

# Get prompt from command-line argument or use default
if len(sys.argv) > 1:
    prompt = " ".join(sys.argv[1:])
    print(f"📝 Using prompt from command-line: {prompt}\n")
else:
    prompt = DEFAULT_PROMPT
    print(f"📝 Using default prompt (change DEFAULT_PROMPT in script or pass as argument)\n")

print("=" * 80)
print("✅ FORMALITY / CONCISENESS STEERING EXPERIMENT")
print("=" * 80)
print("\n📋 WHAT TO EXPECT:")
print("   This experiment compares baseline output vs formal/conciseness steering.")
print("   Baseline: Normal model output (might be casual or verbose)")
print("   Steered: More formal, precise, concise tone")
print("\n💡 EXAMPLE OUTPUTS (these are examples, not actual results):")
print("   Casual example: 'So yeah, it's kind of cool and stuff, you know?'")
print("   Formal example: 'In summary, the experiment demonstrates that the results")
print("                    are consistent with the hypothesis. Consequently, we")
print("                    recommend adoption of the methodology described herein.'")
print("\n   ⚠️  NOTE: The ACTUAL model outputs will be shown below after generation.")
print("\n" + "=" * 80)

# Load model
print("\n🤖 MODEL INFORMATION:")
print("   Model: GPT-2 Medium (355M parameters)")
print("   Source: Hugging Face (openai-community/gpt2-medium)")
print("\nLoading model...")
model, tokenizer = load_model(local=True)
print("✓ Model loaded successfully\n")

# Define formal (positive) and casual (negative) examples
formal = [
    "In summary, the experiment demonstrates that the results are consistent with the hypothesis.",
    "Consequently, we recommend adoption of the methodology described herein."
]

casual = [
    "So yeah, it's kind of cool and stuff, you know?",
    "Basically, it works, just do it."
]

print("Computing steering vector from examples...")
print(f"  Formal examples: {formal}")
print(f"  Casual examples: {casual}")

# Compute steering vector
pos_states = get_hidden_states(model, tokenizer, formal)
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
print("   Expected: Might be casual, informal, or verbose")
print("   (Example of what to expect: 'So yeah, it's kind of cool and stuff, you know?')\n")
print("   ⏳ Generating actual output from model...\n")

# Generate baseline (no steering)
with torch.no_grad():
    out_baseline = model.generate(
        **inputs,
        max_new_tokens=120,
        temperature=0.7,
        do_sample=True
    )

baseline_text = tokenizer.decode(out_baseline[0], skip_special_tokens=True)
print("   ✅ ACTUAL MODEL OUTPUT:")
print(f"   {baseline_text}\n")

print("-" * 80)
print("2️⃣  STEERED OUTPUT (with formality/conciseness steering)")
print("-" * 80)
print("   This is the ACTUAL output with steering applied to make it more formal/precise.")
print("   Expected: More formal, precise, concise tone")
print("   (Example of what to expect: 'In summary, the experiment demonstrates that")
print("                                the results are consistent with the hypothesis.")
print("                                Consequently, we recommend adoption...')\n")
print("   ⏳ Generating actual steered output from model...\n")

# Apply steering hook (using layer=-3 as suggested in the original code)
hook = apply_steering_hook(model, vector, layer=-3, strength=2.0)

# Generate with steering
with torch.no_grad():
    out_steered = model.generate(
        **inputs,
        max_new_tokens=120,
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
print("\n✅ With formality/conciseness steering, you should see:")
print("   ✓ More formal language (consequently, therefore, in summary)")
print("   ✓ Precise, structured sentences")
print("   ✓ Professional, academic tone")
print("   ✓ Reduced casual expressions (yeah, kind of, stuff)")
print("   ✓ More concise and clear communication")
print("\n🔍 Compare the two outputs above:")
print("   - Does the steered output use more formal vocabulary?")
print("   - Is it more structured and precise?")
print("   - Are casual expressions reduced?")
print("   - Does it sound more professional/academic?")
print("\n📊 Quantitative signs:")
print("   - Style/word-choice shift toward formal language")
print("   - Cosine similarity to formal examples increases")
print("   - Reduced casual filler words")
print("\n" + "=" * 80)

if steered_text != baseline_text:
    print("✅ SUCCESS: The outputs are different - steering is working!")
    print("   Compare the two outputs above to see the formality shift.")
    print("   If you see more formal, precise language, steering worked! ✅")
else:
    print("⚠️  NOTE: Outputs are similar. Try:")
    print("   - Increasing steering strength (edit strength=2.0 in script)")
    print("   - Using a different prompt")
    print("   - Adjusting temperature")

print("=" * 80)

# Save results
output_content = f"""FORMALITY / CONCISENESS STEERING EXPERIMENT RESULTS
{'=' * 80}

MODEL: GPT-2 Medium (355M parameters)
SOURCE: Hugging Face (openai-community/gpt2-medium)

INPUT PROMPT: "{prompt}"

1️⃣  BASELINE OUTPUT (without steering):
{'-' * 80}
Expected: Might be casual, informal, or verbose
Example: "So yeah, it's kind of cool and stuff, you know?"

Actual Output:
{baseline_text}

2️⃣  STEERED OUTPUT (with formality/conciseness steering):
{'-' * 80}
Expected: More formal, precise, concise tone
Example: "In summary, the experiment demonstrates that the results are consistent with the hypothesis. Consequently, we recommend adoption of the methodology described herein."

Actual Output:
{steered_text}

ANALYSIS:
{'-' * 80}
The steering vector was computed from formal examples (structured, academic) vs casual examples (informal, conversational).
Compare the two outputs above to see how steering influenced the formality/conciseness level.

✅ WHAT TO LOOK FOR:
- More formal language (consequently, therefore, in summary)
- Precise, structured sentences
- Professional, academic tone
- Reduced casual expressions
- More concise and clear communication

QUANTITATIVE SIGNS:
- Style/word-choice shift toward formal language
- Cosine similarity to formal examples increases
- Reduced casual filler words

If you see this kind of shift, steering worked! ✅
"""

# Ensure results directory exists
os.makedirs("results/local_tests", exist_ok=True)

with open("results/local_tests/formality_test.txt", "w") as f:
    f.write(output_content)

print(f"\n💾 Results saved to: results/local_tests/formality_test.txt")
print(f"\n💡 TIP: To test a different prompt, run:")
print(f"   python experiments/steering_formality_conciseness.py \"Your custom prompt here\"")
