#!/usr/bin/env python3
"""
Simple test script to verify model loading and basic inference.
"""

import sys
from pathlib import Path

try:
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration
except ImportError:
    print("❌ Please install: pip install torch transformers")
    sys.exit(1)

FINAL_MODEL_DIR = Path("final_model")

if not FINAL_MODEL_DIR.exists():
    print(f"❌ {FINAL_MODEL_DIR} not found. Extract final_model.zip first.")
    sys.exit(1)

print("🔄 Loading model...")
try:
    tokenizer = AutoTokenizer.from_pretrained(str(FINAL_MODEL_DIR), use_fast=True)
    model = T5ForConditionalGeneration.from_pretrained(str(FINAL_MODEL_DIR))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"✅ Model loaded successfully!")
    print(f"   Device: {device}")
    print(f"   Parameters: {model.num_parameters():,}")
except Exception as e:
    print(f"❌ Error: {e}")
    sys.exit(1)

# Test translation
print("\n🧪 Testing translation...")
python_code = "def sum_upto(n):\n    s = 0\n    for i in range(n + 1):\n        s += i\n    return s"

text = f"translate Python to C++: {python_code}"
inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    out_ids = model.generate(**inputs, num_beams=4, max_length=512)

cpp_output = tokenizer.decode(out_ids[0], skip_special_tokens=True)

print("\n--- Python Input ---")
print(python_code)
print("\n--- Generated C++ ---")
print(cpp_output)
print("\n✅ Test complete!")
