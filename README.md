# 🧠 LLM Steering Project

<div align="center">

**Activation Steering for Language Models using GPT-2**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.10.0-orange.svg)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/Transformers-4.39.3-green.svg)](https://huggingface.co/transformers/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Control and steer language model behavior through activation steering*

[Features](#-features) • [Installation](#-installation) • [Quick Start](#-quick-start) • [Architecture](#-architecture) • [Experiments](#-experiments)

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Experiments](#-experiments)
- [Results](#-results)
- [Model Information](#-model-information)
- [Project Structure](#-project-structure)
- [Contributing](#-contributing)

---

## 🎯 Overview

This project implements **activation steering** techniques for language models, allowing you to control model behavior by injecting steering vectors into hidden states. We use **GPT-2** (124M parameters, small) to demonstrate steering across different dimensions:

- 🎭 **Emotion Steering** - Shift output towards positive/optimistic tone
- 👤 **Persona Steering** - Control writing style (formal vs casual)
- 🙏 **Politeness Steering** - Reduce toxicity, increase politeness
- 📝 **Formality Steering** - Adjust formality and conciseness

---

## ✨ Features

- ✅ **Easy Setup** - Auto-installation of dependencies and model
- ✅ **Multiple Experiments** - 4 different steering experiments
- ✅ **Clear Results** - Baseline vs steered output comparison
- ✅ **Fast Testing** - Simple test script for quick verification
- ✅ **Well Documented** - Clear code structure and comments
- ✅ **Production Ready** - Proper error handling and validation

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    GPT-2 Model (Small)                     │
│                  (124M Parameters)                         │
│              Hugging Face: gpt2                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Hidden States Extraction                        │
│  ┌──────────────┐         ┌──────────────┐                 │
│  │ Positive     │         │ Negative     │                 │
│  │ Examples     │────────▶│ Examples     │                 │
│  └──────────────┘         └──────────────┘                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│            Steering Vector Computation                       │
│         vector = mean(pos_states) - mean(neg_states)        │
│         vector = vector / norm(vector)                       │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Forward Hook Injection                          │
│    hidden_states = hidden_states + strength * vector         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Steered Text Generation                         │
│         (Model output with steering applied)                 │
└─────────────────────────────────────────────────────────────┘
```

### How It Works

1. **Extract Hidden States** - Get hidden states from positive and negative examples
2. **Compute Steering Vector** - Calculate difference vector between positive and negative states
3. **Inject via Hook** - Register forward hook to inject vector during generation
4. **Generate Text** - Model generates text with steering applied
5. **Compare Results** - Compare baseline vs steered outputs

---

## 📦 Installation

### Prerequisites

- **Python** 3.9 or higher
- **pip** package manager

### Step 1: Clone Repository

```bash
git clone https://github.com/itsparsh10/LLM-Steering-Project.git
cd LLM-Steering-Project
```

### Step 2: Install Dependencies

#### Option A: Install from requirements.txt (Recommended)

```bash
python3 -m pip install -r requirements.txt --user
```

#### Option B: Install Core Dependencies Only

```bash
python3 -m pip install transformers torch tokenizers --user
```

### Step 3: Verify Installation

```bash
python3 -c "import transformers, torch, tokenizers; print('✅ All dependencies installed')"
```

---

## 🚀 Quick Start

### Method 1: One-Command Test (Recommended)

Run the simple test script - it handles everything automatically:

```bash
python3 test_steering_simple.py
```

**What happens:**
- ✅ Checks and installs dependencies if needed
- ✅ Downloads GPT-2 model if needed (~500MB, first run only)
- ✅ Runs emotion steering test
- ✅ Saves results to `results/local_tests/simple_test_result.txt`

### Method 2: Step-by-Step Setup

#### Step 1: Activate Virtual Environment (Optional but Recommended)

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate  # On macOS/Linux
# OR
venv\Scripts\activate     # On Windows
```

#### Step 2: Install Requirements

```bash
pip install -r requirements.txt
```

#### Step 3: Load/Install GPT-2 Model

```bash
python3 ai_model.py
```

This will:
- ✅ Verify dependencies
- ✅ Download GPT-2 model from Hugging Face (~500MB)
- ✅ Verify model loading
- ⏱️ Takes 5-10 minutes on first run

#### Step 4: Run Test Script

```bash
python3 test_steering_simple.py
```

#### Step 5: View Results

```bash
cat results/local_tests/simple_test_result.txt
```

---

## 🧪 Experiments

### Available Experiments

| Experiment | File | Description | Output File |
|------------|------|-------------|-------------|
| 🎭 **Emotion** | `steering_emotion.py` | Positive emotion steering | `emotion_test.txt` |
| 👤 **Persona** | `steering_persona.py` | Teacher-style vs casual | `persona_test.txt` |
| 🙏 **Politeness** | `steering_politeness_or_toxicity_reduction.py` | Politeness vs toxicity | `politeness_test.txt` |
| 📝 **Formality** | `steering_formality_conciseness.py` | Formal vs casual | `formality_test.txt` |

### Running Experiments

#### Run Individual Experiment

```bash
# Emotion steering
python3 experiments/steering_emotion.py

# Persona steering
python3 experiments/steering_persona.py

# Politeness steering
python3 experiments/steering_politeness_or_toxicity_reduction.py

# Formality steering
python3 experiments/steering_formality_conciseness.py
```

#### Run with Custom Prompt

```bash
python3 experiments/steering_emotion.py "Your custom prompt here"
```

### Experiment Details

#### 🎭 Emotion Steering

**Purpose**: Shift output towards positive, optimistic tone

**Examples Used**:
- **Positive**: "I feel extremely happy today.", "Life feels exciting and meaningful."
- **Neutral**: "I am writing a sentence.", "This is a neutral statement."

**Expected Result**: More positive vocabulary, optimistic tone, emotional language

#### 👤 Persona Steering

**Purpose**: Control writing style (structured/educational vs casual)

**Examples Used**:
- **Teacher**: "Let me explain this concept step by step.", "This topic requires careful reasoning."
- **Casual**: "Yeah bro, it's pretty chill.", "Not a big deal honestly."

**Expected Result**: More structured explanations, educational tone, professional language

#### 🙏 Politeness Steering

**Purpose**: Reduce toxicity, increase politeness and constructiveness

**Examples Used**:
- **Polite**: "Please, could you help me with this issue? I would appreciate any assistance."
- **Rude**: "You're an idiot if you can't get this.", "That's stupid, why would you even try that?"

**Expected Result**: More polite language, constructive feedback, reduced toxicity

#### 📝 Formality Steering

**Purpose**: Adjust formality and conciseness level

**Examples Used**:
- **Formal**: "In summary, the experiment demonstrates that the results are consistent with the hypothesis."
- **Casual**: "So yeah, it's kind of cool and stuff, you know?", "Basically, it works, just do it."

**Expected Result**: More formal vocabulary, structured sentences, professional tone

---

## 📊 Results

### Result Files Location

All results are saved in `results/local_tests/`:

```
results/local_tests/
├── simple_test_result.txt          # From test_steering_simple.py
├── emotion_test.txt                 # From steering_emotion.py
├── persona_test.txt                # From steering_persona.py
├── politeness_test.txt              # From steering_politeness_or_toxicity_reduction.py
└── formality_test.txt              # From steering_formality_conciseness.py
```

### Result File Format

Each result file contains:

```
EXPERIMENT RESULTS
================================================================================

MODEL: GPT-2 (124M parameters)
INPUT PROMPT: "..."

1️⃣  BASELINE OUTPUT (without steering):
--------------------------------------------------------------------------------
[Baseline text output]

2️⃣  STEERED OUTPUT (with steering):
--------------------------------------------------------------------------------
[Steered text output]

ANALYSIS:
--------------------------------------------------------------------------------
[Comparison and analysis]
```

### Viewing Results

```bash
# View all results
ls -lh results/local_tests/

# View specific result
cat results/local_tests/simple_test_result.txt
cat results/local_tests/emotion_test.txt
```

---

## 🤖 Model Information

### GPT-2 (Small)

| Property | Value |
|----------|-------|
| **Model Name** | GPT-2 |
| **Model ID** | `gpt2` |
| **Parameters** | 124M |
| **Source** | Hugging Face (`openai-community/gpt2`) |
| **License** | Modified MIT License |
| **Size** | ~500MB (downloads automatically) |
| **Language** | English |
| **Architecture** | Transformer-based language model |
| **Context Length** | 1024 tokens |
| **Vocabulary** | 50,257 tokens |

### Model Card

- **Research Paper**: [Language Models are Unsupervised Multitask Learners](https://arxiv.org/abs/1910.09700)
- **Hugging Face**: [gpt2](https://huggingface.co/openai-community/gpt2)
- **OpenAI Blog**: [GPT-2 Blog Post](https://openai.com/research/better-language-models)

---

## 📁 Project Structure

```
LLM-Steering-Project/
│
├── 📂 core/                                    # Core modules
│   ├── model_loader.py                        # Load GPT-2 model
│   ├── hidden_states.py                       # Extract hidden states from model
│   ├── steering.py                            # Compute and apply steering vectors
│   └── auto_setup.py                          # Auto-setup helper
│
├── 📂 experiments/                             # Experiment scripts
│   ├── steering_emotion.py                    # 🎭 Emotion steering experiment
│   ├── steering_persona.py                    # 👤 Persona steering experiment
│   ├── steering_politeness_or_toxicity_reduction.py  # 🙏 Politeness steering
│   └── steering_formality_conciseness.py      # 📝 Formality steering
│
├── 📂 results/                                 # Results directory
│   └── local_tests/                            # Test results saved here
│       ├── simple_test_result.txt              # Simple test results
│       ├── emotion_test.txt                    # Emotion experiment results
│       ├── persona_test.txt                    # Persona experiment results
│       ├── politeness_test.txt                 # Politeness experiment results
│       └── formality_test.txt                  # Formality experiment results
│
├── 📄 test_steering_simple.py                  # ⭐ Simple test script (recommended)
├── 📄 ai_model.py                              # Model setup script
├── 📄 install_and_test.sh                      # Automated install & test script
├── 📄 requirements.txt                         # Python dependencies
├── 📄 .gitignore                               # Git ignore rules
└── 📄 README.md                                # This file
```

---

## 🛠️ Usage Examples

### Example 1: Quick Test

```bash
# Run simple test
python3 test_steering_simple.py

# View results
cat results/local_tests/simple_test_result.txt
```

### Example 2: Run Emotion Experiment

```bash
# Run with default prompt
python3 experiments/steering_emotion.py

# Run with custom prompt
python3 experiments/steering_emotion.py "What makes you happy?"
```

### Example 3: Run All Experiments

```bash
# Run each experiment
python3 experiments/steering_emotion.py
python3 experiments/steering_persona.py
python3 experiments/steering_politeness_or_toxicity_reduction.py
python3 experiments/steering_formality_conciseness.py

# View all results
cat results/local_tests/*.txt
```

---

## 🔧 Troubleshooting

### Issue: ModuleNotFoundError

**Solution**: Ensure you're in the project directory and dependencies are installed:
```bash
cd LLM-Steering-Project
python3 -m pip install -r requirements.txt --user
```

### Issue: Model Download Slow

**Solution**: Normal on first run. Model is ~500MB. Subsequent runs use cached model.

### Issue: Out of Memory

**Solution**: GPT-2 uses ~1GB RAM. Close other applications if needed.

### Issue: Import Errors

**Solution**: Make sure you're running from project root:
```bash
cd LLM-Steering-Project
python3 test_steering_simple.py
```

---

## 📈 Performance

### Runtime Estimates

| Task | First Run | Subsequent Runs |
|------|-----------|----------------|
| **Model Download** | 5-10 minutes | N/A (cached) |
| **Model Loading** | 30-60 seconds | 10-20 seconds |
| **Experiment Run** | 2-5 minutes | 2-5 minutes |
| **Total (First Run)** | ~10-15 minutes | - |
| **Total (Cached)** | - | ~3-6 minutes |

### System Requirements

- **RAM**: ~2GB minimum (model uses ~1GB)
- **Disk**: ~1GB for model cache
- **CPU**: Any modern CPU (GPU optional, not required)

---

## 🧩 Core Components

### `core/model_loader.py`
Loads GPT-2 model from Hugging Face with proper device handling.

### `core/hidden_states.py`
Extracts hidden states from model layers for steering vector computation.

### `core/steering.py`
Computes steering vectors and applies them via forward hooks.

### `core/auto_setup.py`
Automatically checks and installs dependencies.

---

## 🎓 Understanding Steering

### What is Activation Steering?

Activation steering is a technique to control language model behavior by:
1. Extracting hidden states from positive/negative examples
2. Computing a steering vector (difference between positive and negative)
3. Injecting this vector into model activations during generation
4. Observing the resulting behavioral shift

### Why It Works

Language models encode semantic information in their hidden states. By shifting these states in a specific direction, we can influence the model's output without retraining.

---

## 📚 References

- **Research Paper**: [Language Models are Unsupervised Multitask Learners](https://arxiv.org/abs/1910.09700)
- **Hugging Face Model**: [GPT-2](https://huggingface.co/openai-community/gpt2)
- **OpenAI Blog**: [Better Language Models](https://openai.com/research/better-language-models)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project uses GPT-2 model which is licensed under **Modified MIT License**.

---

## 🙏 Acknowledgments

- **OpenAI** for creating GPT-2
- **Hugging Face** for providing the model and transformers library
- **Research Community** for activation steering techniques

---

<div align="center">

**Made with ❤️ for LLM Research**

[⬆ Back to Top](#-llm-steering-project)

</div>
