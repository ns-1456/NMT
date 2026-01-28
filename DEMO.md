# 🚀 Model Demo Guide

## Quick Demo (Run in Notebook)

Add this cell to your notebook or run in Python:

```python
import inference

# Point to the extracted model
inference.FINAL_MODEL_DIR = Path("final_model")

# Example 1: Sum function
python_code = """def sum_upto(n):
    s = 0
    for i in range(n + 1):
        s += i
    return s"""

print("=== Python Input ===")
print(python_code)
print("\n=== Generated C++ ===")
inference.translate(python_code)
```

## Multiple Examples

```python
import inference
from pathlib import Path

inference.FINAL_MODEL_DIR = Path("final_model")

examples = [
    ("Sum Function", """def sum_upto(n):
    s = 0
    for i in range(n + 1):
        s += i
    return s"""),
    
    ("Find Maximum", """def find_max(arr):
    if not arr:
        return None
    max_val = arr[0]
    for val in arr:
        if val > max_val:
            max_val = val
    return max_val"""),
    
    ("String Reversal", """def reverse_string(s):
    result = ""
    for i in range(len(s) - 1, -1, -1):
        result += s[i]
    return result"""),
]

for name, code in examples:
    print(f"\n{'='*60}")
    print(f"Example: {name}")
    print(f"{'='*60}")
    print("\nPython:")
    print(code)
    print("\nGenerated C++:")
    inference.translate(code)
    print()
```

## Running Locally

If you have PyTorch and transformers installed:

```bash
# Install dependencies
pip install torch transformers

# Run demo
python3 demo.py
```

## Expected Output Format

The model generates C++ code like:

```cpp
int sum_upto(int n) {
    int s = 0;
    for (int i = 0; i <= n; i++) {
        s += i;
    }
    return s;
}
```

## Model Performance

- **Training**: Completed 30 epochs successfully
- **Final Eval Loss**: 0.642
- **Best Model**: Saved at epoch 30
- **Stability**: No training collapses (gradient clipping worked!)
