#!/usr/bin/env python3
"""
Simple test script to verify model loading and basic inference.
"""

import os
import sys
from pathlib import Path

# Force transformers backend to PyTorch explicitly (avoid TensorFlow backend issues)
# Must be set before importing transformers
os.environ["TRANSFORMERS_BACKEND"] = "pt"
# Prevent TensorFlow from being auto-detected/imported
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Workaround: Prevent TensorFlow from being imported by transformers
# This is needed if TensorFlow is installed but not properly configured
class TensorFlowBlocker:
    """Prevent TensorFlow import to avoid backend conflicts."""
    def __getattr__(self, name):
        raise ImportError("TensorFlow is blocked. Use PyTorch backend instead.")

# Block TensorFlow before transformers tries to import it
if "tensorflow" not in sys.modules:
    sys.modules["tensorflow"] = TensorFlowBlocker()

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
# Filter out token_type_ids (T5 doesn't use it)
inputs = {k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}

with torch.no_grad():
    out_ids = model.generate(
        **inputs,
        num_beams=5,  # Increased beams for better quality
        max_length=384,  # Increased to allow longer outputs (trained with max_target_len=512)
        min_length=10,    # Minimum output length
        repetition_penalty=2.0,  # Increased penalty to strongly penalize repetition
        length_penalty=1.2,     # Slight preference for longer sequences
        early_stopping=True,    # Stop when EOS is found
        eos_token_id=tokenizer.eos_token_id or 2,  # Stop token
        pad_token_id=tokenizer.pad_token_id or 1,
        no_repeat_ngram_size=4,  # Prevent 4-gram repetition (more aggressive)
        do_sample=False,  # Use deterministic beam search
        num_return_sequences=1,  # Return only the best sequence
    )

# Manually truncate at EOS token as a safeguard (even though early_stopping should handle this)
eos_id = tokenizer.eos_token_id or 2
output_ids = out_ids[0].cpu().tolist()
if eos_id in output_ids:
    eos_idx = output_ids.index(eos_id)
    output_ids = output_ids[:eos_idx]

cpp_output = tokenizer.decode(output_ids, skip_special_tokens=True)

print("\n--- Python Input ---")
print(python_code)
print("\n--- Generated C++ ---")
print(cpp_output)
print("\n✅ Test complete!")
