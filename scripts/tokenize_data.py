#!/usr/bin/env python3
"""Script to tokenize data with BPE or Unigram tokenizers"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse

# TODO: Import necessary modules from src
# from src.utils.config import load_config
# from src.tokenization.bpe import BPETokenizer
# from src.tokenization.unigram import UnigramTokenizer


def main():
    parser = argparse.ArgumentParser(description="Tokenize dataset with BPE or Unigram")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--tokenizer-type", type=str, choices=["bpe", "unigram"], required=True,
                       help="Type of tokenizer to use")
    parser.add_argument("--vocab-size", type=int, default=None, help="Vocabulary size (overrides config)")
    args = parser.parse_args()
    
    # TODO: Implement tokenizer training script
    # 1. Load config
    # 2. Get vocab size
    # 3. Create tokenizers (BPE or Unigram)
    # 4. Train tokenizers on training data
    # 5. Save tokenizers
    pass


if __name__ == "__main__":
    main()
