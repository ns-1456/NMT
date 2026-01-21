#!/usr/bin/env python3
"""Script to tokenize data with BPE or Unigram tokenizers"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

from src.utils.config import load_config
from src.tokenization.bpe import BPETokenizer
from src.tokenization.unigram import UnigramTokenizer


def main():
    parser = argparse.ArgumentParser(description="Tokenize dataset with BPE or Unigram")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--tokenizer-type", type=str, choices=["bpe", "unigram"], required=True,
                       help="Type of tokenizer to use")
    parser.add_argument("--vocab-size", type=int, default=None, help="Vocabulary size (overrides config)")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Get vocab size
    vocab_size = args.vocab_size or config['tokenization']['vocab_size']
    
    # Data paths
    splits_dir = Path(config['paths']['splits_dir'])
    train_source = splits_dir / "train.source"
    train_target = splits_dir / "train.target"
    
    if not train_source.exists() or not train_target.exists():
        print("Error: Training data not found. Please prepare data splits first.")
        return
    
    # Create tokenizers
    print(f"Training {args.tokenizer_type.upper()} tokenizers...")
    
    if args.tokenizer_type == "bpe":
        source_tokenizer = BPETokenizer(vocab_size=vocab_size)
        target_tokenizer = BPETokenizer(vocab_size=vocab_size)
    else:  # unigram
        source_tokenizer = UnigramTokenizer(vocab_size=vocab_size)
        target_tokenizer = UnigramTokenizer(vocab_size=vocab_size)
    
    # Train source tokenizer
    print(f"Training source tokenizer (vocab_size={vocab_size})...")
    source_tokenizer.train([str(train_source)], vocab_size=vocab_size)
    
    # Train target tokenizer
    print(f"Training target tokenizer (vocab_size={vocab_size})...")
    target_tokenizer.train([str(train_target)], vocab_size=vocab_size)
    
    # Save tokenizers
    if args.tokenizer_type == "bpe":
        source_path = splits_dir / f"source_tokenizer_{args.tokenizer_type}.json"
        target_path = splits_dir / f"target_tokenizer_{args.tokenizer_type}.json"
    else:
        source_path = splits_dir / f"source_tokenizer_{args.tokenizer_type}.model"
        target_path = splits_dir / f"target_tokenizer_{args.tokenizer_type}.model"
    
    print(f"Saving source tokenizer to {source_path}...")
    source_tokenizer.save(source_path)
    
    print(f"Saving target tokenizer to {target_path}...")
    target_tokenizer.save(target_path)
    
    print("Tokenizers trained and saved successfully!")


if __name__ == "__main__":
    main()
