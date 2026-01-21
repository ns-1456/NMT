# English-Gujarati Neural Machine Translation

A low-resource Neural Machine Translation (NMT) system for English-to-Gujarati translation, focusing on parameter efficiency and knowledge distillation.

## Project Overview

This project implements a Transformer-based NMT system with the following key features:

- **Custom Sub-word Tokenization**: Comparison of BPE (Byte Pair Encoding) and Unigram tokenizers
- **Small Transformer Architecture**: Student model with <50M parameters
- **Knowledge Distillation**: Transfer knowledge from a larger teacher model to the student
- **Low-Resource Focus**: Optimized for English-Gujarati, a low-resource language pair

## Project Structure

```
NMT/
├── README.md                 # This file
├── requirements.txt          # Python dependencies
├── config.yaml               # Configuration file
├── data/
│   ├── raw/                  # Raw dataset files
│   ├── processed/            # Preprocessed data
│   └── splits/               # Train/val/test splits
├── src/
│   ├── data/                 # Data loading and preprocessing
│   ├── tokenization/         # BPE and Unigram tokenizers
│   ├── models/               # Transformer and distillation
│   ├── training/             # Training and evaluation
│   └── utils/                # Utilities
├── scripts/
│   ├── train.py              # Main training script
│   ├── evaluate.py           # Evaluation script
│   ├── tokenize_data.py      # Tokenize dataset
│   └── compare_tokenizers.py # Compare tokenizers
├── notebooks/
│   └── analysis.ipynb        # Exploratory analysis
└── checkpoints/              # Model checkpoints
```

## Installation

1. **Clone the repository** (if applicable) or navigate to the project directory.

2. **Create a virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

## Dataset Setup

### Option 1: FLORES Dataset (Recommended)

The FLORES-200 dataset is automatically downloadable. The system will download it when you run the data preparation script.

### Option 2: Custom Dataset

If you have your own English-Gujarati parallel corpus:

1. Place your files in `data/raw/`:
   - `en.txt` - English sentences (one per line)
   - `gu.txt` - Gujarati sentences (one per line)

2. Run preprocessing:
```python
from src.data.preprocess import preprocess_parallel_files
from pathlib import Path

preprocess_parallel_files(
    source_file=Path("data/raw/en.txt"),
    target_file=Path("data/raw/gu.txt"),
    output_source=Path("data/processed/en.txt"),
    output_target=Path("data/processed/gu.txt"),
    min_length=3,
    max_length=128
)
```

3. Create train/val/test splits:
```python
from src.data.dataset import create_data_splits

create_data_splits(
    source_file=Path("data/processed/en.txt"),
    target_file=Path("data/processed/gu.txt"),
    output_dir=Path("data/splits"),
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1
)
```

## Configuration

Edit `config.yaml` to customize:

- **Model architecture**: Adjust layers, dimensions, heads for student/teacher models
- **Training hyperparameters**: Batch size, learning rate, epochs, etc.
- **Tokenization**: Choose BPE or Unigram, set vocabulary sizes
- **Distillation**: Enable/disable, set temperature and alpha

## Usage

### 1. Prepare Data

First, download and prepare the dataset:

```python
from src.data.download import prepare_dataset
from src.data.preprocess import preprocess_parallel_files
from src.data.dataset import create_data_splits
from pathlib import Path

# Download FLORES dataset
en_file, gu_file = prepare_dataset(Path("data"), dataset_name="flores")

# Preprocess
preprocess_parallel_files(
    source_file=en_file,
    target_file=gu_file,
    output_source=Path("data/processed/en.txt"),
    output_target=Path("data/processed/gu.txt"),
    min_length=3,
    max_length=128
)

# Create splits
create_data_splits(
    source_file=Path("data/processed/en.txt"),
    target_file=Path("data/processed/gu.txt"),
    output_dir=Path("data/splits")
)
```

### 2. Train Tokenizers

Train BPE or Unigram tokenizers:

```bash
# Train BPE tokenizers
python scripts/tokenize_data.py --tokenizer-type bpe

# Train Unigram tokenizers
python scripts/tokenize_data.py --tokenizer-type unigram
```

### 3. Compare Tokenizers

Compare BPE and Unigram tokenizers on different vocabulary sizes:

```bash
python scripts/compare_tokenizers.py
```

This will generate a comparison report showing OOV rates and rare word coverage.

### 4. Train Teacher Model

First, train the larger teacher model:

```bash
python scripts/train.py --model-type teacher --config config.yaml
```

### 5. Train Student Model

Train the small student model (with or without distillation):

**With Knowledge Distillation**:
1. Update `config.yaml`:
   ```yaml
   distillation:
     enabled: true
     teacher_checkpoint: "checkpoints/best_model.pt"  # Path to teacher checkpoint
   ```

2. Train student:
```bash
python scripts/train.py --model-type student --config config.yaml
```

**Without Distillation** (standard training):
1. Set `distillation.enabled: false` in `config.yaml`
2. Train student:
```bash
python scripts/train.py --model-type student --config config.yaml
```

### 6. Evaluate Model

Evaluate the trained model:

```bash
python scripts/evaluate.py --checkpoint checkpoints/best_model.pt --split test
```

## Model Architecture

### Student Model (<50M parameters)
- **Layers**: 4 encoder + 4 decoder layers
- **Hidden dimension**: 512
- **Attention heads**: 8
- **Feedforward dimension**: 2048
- **Vocabulary**: 16K tokens per language

### Teacher Model
- **Layers**: 6 encoder + 6 decoder layers
- **Hidden dimension**: 512
- **Attention heads**: 8
- **Feedforward dimension**: 2048
- **Vocabulary**: 16K tokens per language

## Knowledge Distillation

The system implements knowledge distillation with:

- **Temperature scaling**: Softens the teacher's probability distribution
- **Weighted loss**: Combines hard loss (ground truth) and soft loss (teacher predictions)
- **Formula**: `Loss = α * hard_loss + (1-α) * soft_loss`

Default parameters:
- Temperature: 4.0
- Alpha: 0.5 (equal weight)

## Tokenization Comparison

The project includes tools to compare BPE and Unigram tokenizers:

- **Vocabulary sizes tested**: 4K, 8K, 16K, 32K
- **Metrics**: OOV rate, rare word coverage
- **Analysis**: Impact on translation quality

## Evaluation Metrics

- **BLEU Score**: Standard metric for translation quality
- **Perplexity**: Language modeling metric
- **Model Size**: Parameter count and memory usage

## Results

After training, you can generate results for your resume:

- Model size reduction (student vs teacher)
- BLEU score retention (target: 95% of teacher performance)
- Tokenizer comparison results
- Rare word translation performance

Example resume bullet point:
> "Engineered a low-resource NMT system for English-Gujarati; implemented custom BPE tokenization and Knowledge Distillation to reduce model size by 40% while retaining 95% of BLEU score performance."

## Troubleshooting

### CUDA Out of Memory
- Reduce batch size in `config.yaml`
- Reduce model dimensions
- Use gradient accumulation

### Tokenizer Training Fails
- Ensure data files exist in `data/splits/`
- Check file encoding (should be UTF-8)
- Verify sufficient disk space

### Low BLEU Scores
- Increase training epochs
- Adjust learning rate
- Try different vocabulary sizes
- Check data quality and preprocessing

## File Descriptions

- **`src/data/download.py`**: Dataset download utilities
- **`src/data/preprocess.py`**: Text cleaning and normalization
- **`src/data/dataset.py`**: PyTorch Dataset classes
- **`src/tokenization/bpe.py`**: BPE tokenizer implementation
- **`src/tokenization/unigram.py`**: Unigram tokenizer implementation
- **`src/models/transformer.py`**: Transformer architecture
- **`src/models/distillation.py`**: Knowledge distillation loss
- **`src/training/trainer.py`**: Training loop and checkpointing
- **`src/training/evaluator.py`**: BLEU evaluation and generation
- **`scripts/train.py`**: Main training entry point
- **`scripts/evaluate.py`**: Model evaluation
- **`scripts/tokenize_data.py`**: Tokenizer training
- **`scripts/compare_tokenizers.py`**: Tokenizer comparison

## License

This project is for educational/research purposes.

## References

- Vaswani et al., "Attention Is All You Need" (Transformer architecture)
- Hinton et al., "Distilling the Knowledge in a Neural Network" (Knowledge Distillation)
- Sennrich et al., "Neural Machine Translation of Rare Words with Subword Units" (BPE)
- Kudo & Richardson, "SentencePiece: A simple and language independent subword tokenizer" (Unigram)

## Contact

For questions or issues, please open an issue in the repository.
