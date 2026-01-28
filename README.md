# T5-Nano Python → C++ Translator

A neural machine translation model that translates Python code to C++ using a T5-Nano architecture trained from scratch.

## Model Details

- **Architecture**: T5ForConditionalGeneration (T5-Nano)
- **Parameters**: ~20.7M
- **Training**: Trained from scratch (random initialization) on XLCoST dataset
- **Final Eval Loss**: 0.642 (74.9% improvement from initial 2.55)
- **Training Time**: ~1h 39m (30 epochs)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Extract Model (if needed)

If you have `final_model.zip`:

```bash
unzip final_model.zip
```

### 3. Run Demo

```bash
python3 demo.py
```

Or use the inference module directly:

```python
import inference

python_code = """
def sum_upto(n):
    s = 0
    for i in range(n + 1):
        s += i
    return s
"""

cpp_output = inference.translate(python_code)
print(cpp_output)
```

## Project Structure

```
NMT/
├── final_model/          # Trained model checkpoint
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── ...
├── custom_tokenizer/     # Custom Byte-Level BPE tokenizer
├── data/                 # XLCoST dataset
├── notebooks/            # Training notebooks
├── train.py              # Training script
├── inference.py          # Inference module
├── demo.py               # Demo script with examples
└── model_config.py       # Model architecture config
```

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Learning Rate | 5e-5 |
| Batch Size | 32 |
| Epochs | 30 |
| Max Grad Norm | 1.0 |
| Warmup Steps | 500 |
| Precision | FP16 |

## Training Results

- **Best Eval Loss**: 0.642 (Epoch 30)
- **Training Loss**: 0.802 (final epoch)
- **Total Improvement**: 74.9% reduction in eval loss
- **Stability**: Stable training with gradient clipping (no collapses)

## Example Translations

### Input (Python):
```python
def sum_upto(n):
    s = 0
    for i in range(n + 1):
        s += i
    return s
```

### Output (C++):
[Run `demo.py` to see generated translations]

## Files

- `train.py` - Training script with fixed hyperparameters
- `inference.py` - Simple inference module
- `demo.py` - Comprehensive demo with multiple examples
- `model_config.py` - T5-Nano architecture configuration
- `data_prep.py` - XLCoST dataset preparation

## Notes

- Model trained on Google Colab with T4/A100 GPU
- Uses custom Byte-Level BPE tokenizer (vocab_size=16k)
- Model saved with Git LFS (147 MB compressed)
