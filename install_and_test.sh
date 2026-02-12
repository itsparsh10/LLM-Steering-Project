#!/bin/bash
# Install dependencies, load model, and test script

echo "=================================================================================="
echo "🚀 INSTALLING DEPENDENCIES AND TESTING GPT-2 MEDIUM MODEL"
echo "=================================================================================="

cd "$(dirname "$0")"

echo ""
echo "Step 1: Installing dependencies..."
python3 -m pip install transformers torch tokenizers --user

if [ $? -ne 0 ]; then
    echo "❌ Failed to install dependencies"
    exit 1
fi

echo ""
echo "Step 2: Testing model loading..."
python3 -c "
import sys
sys.path.insert(0, '.')
from core.model_loader import load_model
print('Loading GPT-2 Medium model...')
model, tokenizer = load_model(local=True, verbose=True)
print('✅ Model loaded successfully!')
"

if [ $? -ne 0 ]; then
    echo "❌ Failed to load model"
    exit 1
fi

echo ""
echo "Step 3: Running test script..."
python3 test_steering_simple.py

if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================================================="
    echo "✅ SUCCESS! Test completed."
    echo "=================================================================================="
    echo ""
    echo "📁 Results saved to: results/local_tests/simple_test_result.txt"
    echo ""
    echo "View results with:"
    echo "  cat results/local_tests/simple_test_result.txt"
else
    echo ""
    echo "❌ Test failed"
    exit 1
fi
