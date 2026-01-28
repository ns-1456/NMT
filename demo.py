#!/usr/bin/env python3
"""
Demo script for T5-Nano Python → C++ Translator

Shows multiple examples of Python code being translated to C++.
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

# Check dependencies
try:
    import torch
    from transformers import AutoTokenizer, T5ForConditionalGeneration
except ImportError as e:
    print(f"❌ Missing dependencies: {e}")
    print("\nPlease install requirements:")
    print("  pip install torch transformers")
    sys.exit(1)

TASK_PREFIX = "translate Python to C++: "
FINAL_MODEL_DIR = Path("final_model")


def load_model():
    """Load the trained model and tokenizer."""
    if not FINAL_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"{FINAL_MODEL_DIR} not found. Extract final_model.zip first."
        )
    
    print("🔄 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(FINAL_MODEL_DIR), use_fast=True)
    model = T5ForConditionalGeneration.from_pretrained(str(FINAL_MODEL_DIR))
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded! Device: {device}")
    print(f"   Parameters: {model.num_parameters():,}")
    return model, tokenizer, device


def translate(model, tokenizer, device, python_code: str) -> str:
    """Translate Python code to C++."""
    text = python_code
    if not text.startswith(TASK_PREFIX):
        text = TASK_PREFIX + text
    
    # Tokenize
    inputs = tokenizer(text, return_tensors="pt", truncation=False)
    if inputs.get("input_ids") is not None and inputs["input_ids"].shape[-1] > 256:
        inputs["input_ids"] = inputs["input_ids"][:, :256]
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"][:, :256]
    
    # Filter out token_type_ids (T5 doesn't use it)
    inputs = {k: v.to(device) for k, v in inputs.items() if k != "token_type_ids"}
    
    # Generate with improved parameters to prevent repetition
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
    
    decoded = tokenizer.decode(output_ids, skip_special_tokens=True)
    return decoded


def print_separator():
    print("\n" + "=" * 80 + "\n")


def demo_example(name: str, python_code: str, model, tokenizer, device):
    """Run a single translation demo."""
    print_separator()
    print(f"📝 Example: {name}")
    print("\n--- Python Input ---")
    print(python_code)
    print("\n--- Generated C++ ---")
    cpp_output = translate(model, tokenizer, device, python_code)
    print(cpp_output)
    return cpp_output


def main():
    print("🚀 T5-Nano Python → C++ Translator Demo")
    print("=" * 80)
    
    try:
        model, tokenizer, device = load_model()
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)
    
    # Example 1: Simple function
    demo_example(
        "Sum Function",
        """def sum_upto(n):
    s = 0
    for i in range(n + 1):
        s += i
    return s""",
        model, tokenizer, device
    )
    
    # Example 2: List operations
    demo_example(
        "Find Maximum",
        """def find_max(arr):
    if not arr:
        return None
    max_val = arr[0]
    for val in arr:
        if val > max_val:
            max_val = val
    return max_val""",
        model, tokenizer, device
    )
    
    # Example 3: String manipulation
    demo_example(
        "String Reversal",
        """def reverse_string(s):
    result = ""
    for i in range(len(s) - 1, -1, -1):
        result += s[i]
    return result""",
        model, tokenizer, device
    )
    
    # Example 4: Conditional logic
    demo_example(
        "Even Check",
        """def is_even(n):
    if n % 2 == 0:
        return True
    else:
        return False""",
        model, tokenizer, device
    )
    
    print_separator()
    print("✨ Demo complete!")
    print(f"\nModel Statistics:")
    print(f"  - Parameters: {model.num_parameters():,}")
    print(f"  - Device: {device}")
    print(f"  - Model Type: T5-Nano (20.7M parameters)")


if __name__ == "__main__":
    main()
