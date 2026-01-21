#!/usr/bin/env python3
"""Compare student and teacher models"""

import sys
from pathlib import Path
import json
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.models.transformer import TransformerNMT
from src.tokenization.bpe import BPETokenizer
from src.tokenization.unigram import UnigramTokenizer
from src.training.evaluator import evaluate_model
from src.data.dataset import ParallelDataset, get_dataloader


def load_tokenizers(config, splits_dir):
    """Load tokenizers."""
    tokenizer_type = config['tokenization']['type']
    
    if tokenizer_type == "bpe":
        source_tokenizer = BPETokenizer()
        target_tokenizer = BPETokenizer()
        source_tokenizer.load(splits_dir / f"source_tokenizer_{tokenizer_type}.json")
        target_tokenizer.load(splits_dir / f"target_tokenizer_{tokenizer_type}.json")
    else:
        source_tokenizer = UnigramTokenizer()
        target_tokenizer = UnigramTokenizer()
        source_tokenizer.load(splits_dir / f"source_tokenizer_{tokenizer_type}.model")
        target_tokenizer.load(splits_dir / f"target_tokenizer_{tokenizer_type}.model")
    
    return source_tokenizer, target_tokenizer


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare student and teacher models")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--teacher-checkpoint", type=str, required=True, help="Path to teacher checkpoint")
    parser.add_argument("--student-checkpoint", type=str, required=True, help="Path to student checkpoint")
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test",
                       help="Dataset split to evaluate on")
    args = parser.parse_args()
    
    config = load_config(args.config)
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    
    splits_dir = Path(config['paths']['splits_dir'])
    source_tokenizer, target_tokenizer = load_tokenizers(config, splits_dir)
    
    src_vocab_size = source_tokenizer.get_vocab_size() if hasattr(source_tokenizer, 'get_vocab_size') else 16000
    tgt_vocab_size = target_tokenizer.get_vocab_size() if hasattr(target_tokenizer, 'get_vocab_size') else 16000
    
    # Create dataset
    dataset = ParallelDataset(
        splits_dir / f"{args.split}.source",
        splits_dir / f"{args.split}.target",
        source_tokenizer,
        target_tokenizer,
        max_length=config['dataset']['max_length']
    )
    
    dataloader = get_dataloader(dataset, batch_size=config['training']['batch_size'], shuffle=False)
    
    results = {}
    
    # Evaluate teacher
    print("Evaluating teacher model...")
    teacher_checkpoint = torch.load(args.teacher_checkpoint, map_location=device)
    teacher_config = teacher_checkpoint.get('config', {}).get('model', config['model'])['teacher']
    
    teacher_model = TransformerNMT(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=teacher_config.get('d_model', 512),
        nhead=teacher_config.get('num_heads', 8),
        num_encoder_layers=teacher_config.get('num_layers', 6),
        num_decoder_layers=teacher_config.get('num_layers', 6),
        dim_feedforward=teacher_config.get('d_ff', 2048),
        max_seq_length=teacher_config.get('max_seq_length', 128),
        dropout=teacher_config.get('dropout', 0.1),
        pad_token_id=source_tokenizer.pad_token_id
    )
    teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
    teacher_model = teacher_model.to(device)
    
    teacher_results = evaluate_model(teacher_model, dataloader, target_tokenizer, device)
    teacher_params = teacher_model.count_parameters()
    teacher_size_mb = teacher_model.get_model_size_mb()
    
    results['teacher'] = {
        'bleu': teacher_results['bleu'],
        'loss': teacher_results['loss'],
        'parameters': teacher_params,
        'parameters_millions': teacher_params / 1e6,
        'model_size_mb': teacher_size_mb
    }
    
    # Evaluate student
    print("Evaluating student model...")
    student_checkpoint = torch.load(args.student_checkpoint, map_location=device)
    student_config = student_checkpoint.get('config', {}).get('model', config['model'])['student']
    
    student_model = TransformerNMT(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=student_config.get('d_model', 512),
        nhead=student_config.get('num_heads', 8),
        num_encoder_layers=student_config.get('num_layers', 4),
        num_decoder_layers=student_config.get('num_layers', 4),
        dim_feedforward=student_config.get('d_ff', 2048),
        max_seq_length=student_config.get('max_seq_length', 128),
        dropout=student_config.get('dropout', 0.1),
        pad_token_id=source_tokenizer.pad_token_id
    )
    student_model.load_state_dict(student_checkpoint['model_state_dict'])
    student_model = student_model.to(device)
    
    student_results = evaluate_model(student_model, dataloader, target_tokenizer, device)
    student_params = student_model.count_parameters()
    student_size_mb = student_model.get_model_size_mb()
    
    results['student'] = {
        'bleu': student_results['bleu'],
        'loss': student_results['loss'],
        'parameters': student_params,
        'parameters_millions': student_params / 1e6,
        'model_size_mb': student_size_mb
    }
    
    # Calculate compression metrics
    results['compression'] = {
        'size_reduction_percent': (1 - student_params / teacher_params) * 100,
        'bleu_retention_percent': (student_results['bleu'] / teacher_results['bleu']) * 100 if teacher_results['bleu'] > 0 else 0,
        'bleu_drop': teacher_results['bleu'] - student_results['bleu']
    }
    
    # Print results
    print("\n" + "="*80)
    print("MODEL COMPARISON RESULTS")
    print("="*80)
    print(f"\nTeacher Model:")
    print(f"  Parameters: {results['teacher']['parameters_millions']:.2f}M")
    print(f"  Model Size: {results['teacher']['model_size_mb']:.2f} MB")
    print(f"  BLEU Score: {results['teacher']['bleu']:.2f}")
    print(f"  Loss: {results['teacher']['loss']:.4f}")
    
    print(f"\nStudent Model:")
    print(f"  Parameters: {results['student']['parameters_millions']:.2f}M")
    print(f"  Model Size: {results['student']['model_size_mb']:.2f} MB")
    print(f"  BLEU Score: {results['student']['bleu']:.2f}")
    print(f"  Loss: {results['student']['loss']:.4f}")
    
    print(f"\nCompression Metrics:")
    print(f"  Size Reduction: {results['compression']['size_reduction_percent']:.1f}%")
    print(f"  BLEU Retention: {results['compression']['bleu_retention_percent']:.1f}%")
    print(f"  BLEU Drop: {results['compression']['bleu_drop']:.2f}")
    print("="*80)
    
    # Save results
    output_file = Path(config['paths']['checkpoint_dir']) / "model_comparison.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
