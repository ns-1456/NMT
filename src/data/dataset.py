"""Dataset loading and PyTorch Dataset classes"""

from pathlib import Path
from typing import List, Tuple, Optional
import torch
from torch.utils.data import Dataset, DataLoader
import random


class ParallelDataset(Dataset):
    """PyTorch Dataset for parallel text data.
    
    Args:
        source_file: Path to source language file
        target_file: Path to target language file
        source_tokenizer: Tokenizer for source language
        target_tokenizer: Tokenizer for target language
        max_length: Maximum sequence length
    """
    
    def __init__(
        self,
        source_file: Path,
        target_file: Path,
        source_tokenizer,
        target_tokenizer,
        max_length: int = 128
    ):
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.max_length = max_length
        
        # Load parallel sentences
        with open(source_file, 'r', encoding='utf-8') as f:
            self.source_lines = [line.strip() for line in f if line.strip()]
        
        with open(target_file, 'r', encoding='utf-8') as f:
            self.target_lines = [line.strip() for line in f if line.strip()]
        
        # Ensure same length
        min_len = min(len(self.source_lines), len(self.target_lines))
        self.source_lines = self.source_lines[:min_len]
        self.target_lines = self.target_lines[:min_len]
        
        assert len(self.source_lines) == len(self.target_lines), \
            "Source and target files must have the same number of lines"
    
    def __len__(self) -> int:
        return len(self.source_lines)
    
    def __getitem__(self, idx: int) -> dict:
        """Get a single parallel sentence pair.
        
        Args:
            idx: Index of the sentence pair
            
        Returns:
            Dictionary with tokenized source and target
        """
        source_text = self.source_lines[idx]
        target_text = self.target_lines[idx]
        
        # Tokenize source
        source_result = self.source_tokenizer.encode(
            source_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Handle dict or tensor output
        if isinstance(source_result, dict):
            source_tokens = source_result['input_ids']
        else:
            source_tokens = source_result
        
        # Tokenize target (for input and labels)
        target_result = self.target_tokenizer.encode(
            target_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Handle dict or tensor output
        if isinstance(target_result, dict):
            target_tokens = target_result['input_ids']
        else:
            target_tokens = target_result
        
        # Create decoder input (shifted target) and labels
        decoder_input_ids = target_tokens.clone()
        labels = target_tokens.clone()
        
        # Shift labels for next-token prediction
        # Labels are the target tokens shifted by one position
        labels[labels == self.target_tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': source_tokens.squeeze(0),
            'decoder_input_ids': decoder_input_ids.squeeze(0),
            'labels': labels.squeeze(0),
            'source_text': source_text,
            'target_text': target_text
        }


def create_data_splits(
    source_file: Path,
    target_file: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[Path, Path, Path, Path, Path, Path]:
    """Create train/val/test splits from parallel files.
    
    Args:
        source_file: Path to source language file
        target_file: Path to target language file
        output_dir: Directory to save splits
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data
        test_ratio: Ratio of test data
        seed: Random seed for shuffling
        
    Returns:
        Tuple of (train_source, train_target, val_source, val_target, test_source, test_target) paths
    """
    # Load all lines
    with open(source_file, 'r', encoding='utf-8') as f:
        source_lines = [line.strip() for line in f if line.strip()]
    
    with open(target_file, 'r', encoding='utf-8') as f:
        target_lines = [line.strip() for line in f if line.strip()]
    
    # Ensure same length
    min_len = min(len(source_lines), len(target_lines))
    source_lines = source_lines[:min_len]
    target_lines = target_lines[:min_len]
    
    # Shuffle
    random.seed(seed)
    pairs = list(zip(source_lines, target_lines))
    random.shuffle(pairs)
    source_lines, target_lines = zip(*pairs)
    source_lines = list(source_lines)
    target_lines = list(target_lines)
    
    # Calculate split sizes
    total = len(source_lines)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    # Split
    train_source = source_lines[:train_size]
    train_target = target_lines[:train_size]
    
    val_source = source_lines[train_size:train_size + val_size]
    val_target = target_lines[train_size:train_size + val_size]
    
    test_source = source_lines[train_size + val_size:]
    test_target = target_lines[train_size + val_size:]
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    train_source_path = output_dir / "train.source"
    train_target_path = output_dir / "train.target"
    val_source_path = output_dir / "val.source"
    val_target_path = output_dir / "val.target"
    test_source_path = output_dir / "test.source"
    test_target_path = output_dir / "test.target"
    
    def write_file(path: Path, lines: List[str]):
        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
    
    write_file(train_source_path, train_source)
    write_file(train_target_path, train_target)
    write_file(val_source_path, val_source)
    write_file(val_target_path, val_target)
    write_file(test_source_path, test_source)
    write_file(test_target_path, test_target)
    
    print(f"Created splits: Train={len(train_source)}, Val={len(val_source)}, Test={len(test_source)}")
    
    return (
        train_source_path, train_target_path,
        val_source_path, val_target_path,
        test_source_path, test_target_path
    )


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0
) -> DataLoader:
    """Create a DataLoader for a dataset.
    
    Args:
        dataset: PyTorch Dataset
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
