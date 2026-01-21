#!/usr/bin/env python3
"""Evaluation script for English-Gujarati NMT"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse

from src.utils.config import load_config
from src.data.dataset import ParallelDataset, get_dataloader
from src.tokenization.bpe import BPETokenizer
from src.tokenization.unigram import UnigramTokenizer
from src.models.transformer import TransformerNMT
from src.training.evaluator import evaluate_model


def load_tokenizers(config, splits_dir):
    """Load tokenizers."""
    tokenization_config = config['tokenization']
    tokenizer_type = tokenization_config['type']
    
    # Source tokenizer
    source_tokenizer_path = splits_dir / f"source_tokenizer_{tokenizer_type}.json"
    if tokenizer_type == "bpe":
        source_tokenizer = BPETokenizer()
        if source_tokenizer_path.exists():
            source_tokenizer.load(source_tokenizer_path)
        else:
            raise FileNotFoundError(f"Source tokenizer not found: {source_tokenizer_path}")
    else:  # unigram
        source_tokenizer_path = splits_dir / f"source_tokenizer_{tokenizer_type}.model"
        source_tokenizer = UnigramTokenizer()
        if source_tokenizer_path.exists():
            source_tokenizer.load(source_tokenizer_path)
        else:
            raise FileNotFoundError(f"Source tokenizer not found: {source_tokenizer_path}")
    
    # Target tokenizer
    target_tokenizer_path = splits_dir / f"target_tokenizer_{tokenizer_type}.json"
    if tokenizer_type == "bpe":
        target_tokenizer = BPETokenizer()
        if target_tokenizer_path.exists():
            target_tokenizer.load(target_tokenizer_path)
        else:
            raise FileNotFoundError(f"Target tokenizer not found: {target_tokenizer_path}")
    else:  # unigram
        target_tokenizer_path = splits_dir / f"target_tokenizer_{tokenizer_type}.model"
        target_tokenizer = UnigramTokenizer()
        if target_tokenizer_path.exists():
            target_tokenizer.load(target_tokenizer_path)
        else:
            raise FileNotFoundError(f"Target tokenizer not found: {target_tokenizer_path}")
    
    return source_tokenizer, target_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Evaluate English-Gujarati NMT model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test",
                       help="Dataset split to evaluate on")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Device
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # Data paths
    splits_dir = Path(config['paths']['splits_dir'])
    
    # Load tokenizers
    print("Loading tokenizers...")
    source_tokenizer, target_tokenizer = load_tokenizers(config, splits_dir)
    src_vocab_size = source_tokenizer.get_vocab_size() if hasattr(source_tokenizer, 'get_vocab_size') else (source_tokenizer.vocab_size if hasattr(source_tokenizer, 'vocab_size') else 16000)
    tgt_vocab_size = target_tokenizer.get_vocab_size() if hasattr(target_tokenizer, 'get_vocab_size') else (target_tokenizer.vocab_size if hasattr(target_tokenizer, 'vocab_size') else 16000)
    
    # Create dataset
    print(f"Creating {args.split} dataset...")
    dataset = ParallelDataset(
        splits_dir / f"{args.split}.source",
        splits_dir / f"{args.split}.target",
        source_tokenizer,
        target_tokenizer,
        max_length=config['dataset']['max_length']
    )
    
    # Create dataloader
    dataloader = get_dataloader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    # Load model
    print(f"Loading model from {args.checkpoint}...")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Determine model config from checkpoint
    model_config = checkpoint.get('config', {}).get('model', config['model'])
    if 'student' in model_config:
        model_config = model_config['student']
    elif 'teacher' in model_config:
        model_config = model_config['teacher']
    
    model = TransformerNMT(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=model_config.get('d_model', 512),
        nhead=model_config.get('num_heads', 8),
        num_encoder_layers=model_config.get('num_layers', 4),
        num_decoder_layers=model_config.get('num_layers', 4),
        dim_feedforward=model_config.get('d_ff', 2048),
        max_seq_length=model_config.get('max_seq_length', 128),
        dropout=model_config.get('dropout', 0.1),
        pad_token_id=source_tokenizer.pad_token_id
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Evaluate
    print("Evaluating model...")
    results = evaluate_model(
        model,
        dataloader,
        target_tokenizer,
        device,
        max_length=config.get('evaluation', {}).get('max_length', 128),
        beam_size=config.get('evaluation', {}).get('beam_size', 5)
    )
    
    # Print results
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"BLEU Score: {results['bleu']:.2f}")
    print(f"Loss: {results['loss']:.4f}")
    print("="*50)
    
    # Save predictions
    output_file = Path(config['paths']['checkpoint_dir']) / f"predictions_{args.split}.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for pred, ref in zip(results['predictions'], results['references']):
            f.write(f"PRED: {pred}\n")
            f.write(f"REF:  {ref}\n\n")
    
    print(f"\nPredictions saved to: {output_file}")


if __name__ == "__main__":
    main()
