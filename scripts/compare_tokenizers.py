#!/usr/bin/env python3
"""Script to compare BPE and Unigram tokenizers"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from typing import List, Dict
import json

from src.utils.config import load_config
from src.tokenization.bpe import BPETokenizer
from src.tokenization.unigram import UnigramTokenizer
from src.utils.metrics import analyze_tokenizer_coverage


def load_texts(file_path: Path) -> List[str]:
    """Load texts from file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def compare_tokenizers(
    texts: List[str],
    vocab_sizes: List[int],
    tokenizer_types: List[str]
) -> Dict:
    """Compare tokenizers on different vocabulary sizes.
    
    Args:
        texts: List of texts to analyze
        vocab_sizes: List of vocabulary sizes to test
        tokenizer_types: List of tokenizer types to test
        
    Returns:
        Dictionary with comparison results
    """
    results = {}
    
    for tokenizer_type in tokenizer_types:
        results[tokenizer_type] = {}
        
        for vocab_size in vocab_sizes:
            print(f"\nTraining {tokenizer_type.upper()} tokenizer with vocab_size={vocab_size}...")
            
            # Create temporary file for training
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, encoding='utf-8') as f:
                temp_file = f.name
                f.write('\n'.join(texts))
            
            try:
                # Train tokenizer
                if tokenizer_type == "bpe":
                    tokenizer = BPETokenizer(vocab_size=vocab_size)
                else:  # unigram
                    tokenizer = UnigramTokenizer(vocab_size=vocab_size)
                
                tokenizer.train([temp_file], vocab_size=vocab_size)
                
                # Analyze coverage
                coverage = analyze_tokenizer_coverage(tokenizer, texts[:1000])  # Sample for speed
                
                results[tokenizer_type][vocab_size] = {
                    'oov_rate': coverage['oov_rate'],
                    'rare_word_coverage': coverage['rare_word_coverage'],
                    'total_words': coverage['total_words'],
                    'oov_words': coverage['oov_words'],
                    'rare_words': coverage['rare_words']
                }
                
                print(f"  OOV Rate: {coverage['oov_rate']:.2f}%")
                print(f"  Rare Word Coverage: {coverage['rare_word_coverage']:.2f}%")
                
            finally:
                # Clean up
                import os
                if os.path.exists(temp_file):
                    os.remove(temp_file)
    
    return results


def print_comparison_table(results: Dict):
    """Print comparison results as a table."""
    print("\n" + "="*80)
    print("Tokenizer Comparison Results")
    print("="*80)
    
    # Get all vocab sizes
    vocab_sizes = set()
    for tokenizer_type in results:
        vocab_sizes.update(results[tokenizer_type].keys())
    vocab_sizes = sorted(vocab_sizes)
    
    # Print header
    print(f"\n{'Vocab Size':<12}", end="")
    for tokenizer_type in results:
        print(f"{tokenizer_type.upper():<20}", end="")
    print()
    print("-" * 80)
    
    # Print OOV rates
    print("OOV Rate (%):")
    for vocab_size in vocab_sizes:
        print(f"{vocab_size:<12}", end="")
        for tokenizer_type in results:
            if vocab_size in results[tokenizer_type]:
                oov = results[tokenizer_type][vocab_size]['oov_rate']
                print(f"{oov:>8.2f}%{'':<12}", end="")
            else:
                print(f"{'N/A':<20}", end="")
        print()
    
    # Print rare word coverage
    print("\nRare Word Coverage (%):")
    for vocab_size in vocab_sizes:
        print(f"{vocab_size:<12}", end="")
        for tokenizer_type in results:
            if vocab_size in results[tokenizer_type]:
                coverage = results[tokenizer_type][vocab_size]['rare_word_coverage']
                print(f"{coverage:>8.2f}%{'':<12}", end="")
            else:
                print(f"{'N/A':<20}", end="")
        print()
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description="Compare BPE and Unigram tokenizers")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--data-file", type=str, default=None, help="Path to data file (uses train.source if not specified)")
    parser.add_argument("--vocab-sizes", type=int, nargs="+", default=None,
                       help="Vocabulary sizes to test (overrides config)")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Get vocab sizes
    vocab_sizes = args.vocab_sizes or config['tokenization'].get('vocab_sizes_to_test', [4000, 8000, 16000, 32000])
    
    # Get data file
    if args.data_file:
        data_file = Path(args.data_file)
    else:
        splits_dir = Path(config['paths']['splits_dir'])
        data_file = splits_dir / "train.source"
    
    if not data_file.exists():
        print(f"Error: Data file not found: {data_file}")
        return
    
    # Load texts
    print(f"Loading texts from {data_file}...")
    texts = load_texts(data_file)
    print(f"Loaded {len(texts)} sentences")
    
    # Compare tokenizers
    tokenizer_types = ["bpe", "unigram"]
    results = compare_tokenizers(texts, vocab_sizes, tokenizer_types)
    
    # Print results
    print_comparison_table(results)
    
    # Save results
    output_file = Path(config['paths']['checkpoint_dir']) / "tokenizer_comparison.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
