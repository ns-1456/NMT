#!/usr/bin/env python3
"""Generate comprehensive evaluation report"""

import sys
from pathlib import Path
import json
import torch
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.models.transformer import TransformerNMT
from src.tokenization.bpe import BPETokenizer
from src.tokenization.unigram import UnigramTokenizer


def load_model(checkpoint_path, config, source_tokenizer, target_tokenizer, device, model_type=None):
    """Load model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint
        config: Config dict
        source_tokenizer: Source tokenizer
        target_tokenizer: Target tokenizer
        device: Device to load on
        model_type: Optional, 'teacher' or 'student' to force model type
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    checkpoint_config = checkpoint.get('config', {})
    model_config_dict = checkpoint_config.get('model', config.get('model', {}))
    
    # Determine model type
    if model_type:
        # Use specified model type
        if model_type == 'teacher' and 'teacher' in model_config_dict:
            model_config = model_config_dict['teacher']
        elif model_type == 'student' and 'student' in model_config_dict:
            model_config = model_config_dict['student']
        else:
            # Fallback to config file
            model_config = config.get('model', {}).get(model_type, model_config_dict.get('student', {}))
    else:
        # Try to infer from checkpoint config
        # Check if we can determine from saved model architecture hints
        # Default: assume student if not specified
        if 'student' in model_config_dict:
            model_config = model_config_dict['student']
        elif 'teacher' in model_config_dict:
            model_config = model_config_dict['teacher']
        else:
            # Fallback: use student config from main config
            model_config = config.get('model', {}).get('student', {})
    
    src_vocab_size = source_tokenizer.get_vocab_size() if hasattr(source_tokenizer, 'get_vocab_size') else 16000
    tgt_vocab_size = target_tokenizer.get_vocab_size() if hasattr(target_tokenizer, 'get_vocab_size') else 16000
    
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
    return model, checkpoint


def generate_report(config_path="config.yaml"):
    """Generate comprehensive evaluation report."""
    config = load_config(config_path)
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Load tokenizers
    splits_dir = Path(config['paths']['splits_dir'])
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
    
    report = {
        'timestamp': datetime.now().isoformat(),
        'config': config,
        'models': {}
    }
    
    # Check for teacher model (from config or separate checkpoint)
    teacher_checkpoint_path = config.get('distillation', {}).get('teacher_checkpoint')
    if teacher_checkpoint_path:
        teacher_checkpoint = Path(teacher_checkpoint_path)
        if not teacher_checkpoint.is_absolute():
            teacher_checkpoint = Path.cwd() / teacher_checkpoint
    else:
        # Fallback: check for teacher checkpoint in checkpoint_dir
        teacher_checkpoint = checkpoint_dir / "teacher_best_model.pt"
        if not teacher_checkpoint.exists():
            teacher_checkpoint = None
    
    if teacher_checkpoint and teacher_checkpoint.exists():
        print("Analyzing teacher model...")
        teacher_model, teacher_ckpt = load_model(teacher_checkpoint, config, source_tokenizer, target_tokenizer, device, model_type='teacher')
        teacher_model = teacher_model.to(device)
        teacher_model.eval()
        
        param_count = teacher_model.count_parameters()
        model_size_mb = teacher_model.get_model_size_mb()
        
        report['models']['teacher'] = {
            'checkpoint': str(teacher_checkpoint),
            'parameters': param_count,
            'parameters_millions': param_count / 1e6,
            'model_size_mb': model_size_mb,
            'epoch': teacher_ckpt.get('epoch', 'unknown'),
            'best_bleu': teacher_ckpt.get('best_bleu', 0.0),
            'global_step': teacher_ckpt.get('global_step', 0)
        }
    
    # Check for student model (typically at best_model.pt after student training)
    student_checkpoint = checkpoint_dir / "best_model.pt"
    if student_checkpoint.exists():
        print("Analyzing student model...")
        student_model, student_ckpt = load_model(student_checkpoint, config, source_tokenizer, target_tokenizer, device, model_type='student')
        student_model = student_model.to(device)
        student_model.eval()
        
        param_count = student_model.count_parameters()
        model_size_mb = student_model.get_model_size_mb()
        
        report['models']['student'] = {
            'checkpoint': str(student_checkpoint),
            'parameters': param_count,
            'parameters_millions': param_count / 1e6,
            'model_size_mb': model_size_mb,
            'epoch': student_ckpt.get('epoch', 'unknown'),
            'best_bleu': student_ckpt.get('best_bleu', 0.0),
            'global_step': student_ckpt.get('global_step', 0)
        }
    
    # Load training history if available
    history_path = checkpoint_dir / "training_history.json"
    if history_path.exists():
        with open(history_path, 'r') as f:
            report['training_history'] = json.load(f)
    
    # Load tokenizer comparison if available
    tokenizer_comp_path = checkpoint_dir / "tokenizer_comparison.json"
    if tokenizer_comp_path.exists():
        with open(tokenizer_comp_path, 'r') as f:
            report['tokenizer_comparison'] = json.load(f)
    
    # Save report
    report_path = checkpoint_dir / "evaluation_report.json"
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\nReport generated: {report_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION REPORT SUMMARY")
    print("="*60)
    
    if 'teacher' in report['models']:
        t = report['models']['teacher']
        print(f"\nTeacher Model:")
        print(f"  Parameters: {t['parameters_millions']:.2f}M")
        print(f"  Model Size: {t['model_size_mb']:.2f} MB")
        print(f"  Best BLEU: {t['best_bleu']:.2f}")
    
    if 'student' in report['models']:
        s = report['models']['student']
        print(f"\nStudent Model:")
        print(f"  Parameters: {s['parameters_millions']:.2f}M")
        print(f"  Model Size: {s['model_size_mb']:.2f} MB")
        print(f"  Best BLEU: {s['best_bleu']:.2f}")
        
        if 'teacher' in report['models']:
            t = report['models']['teacher']
            size_reduction = (1 - s['parameters'] / t['parameters']) * 100
            bleu_retention = (s['best_bleu'] / t['best_bleu']) * 100 if t['best_bleu'] > 0 else 0
            print(f"\nCompression:")
            print(f"  Size Reduction: {size_reduction:.1f}%")
            print(f"  BLEU Retention: {bleu_retention:.1f}%")
    
    print("="*60)
    
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate evaluation report")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    generate_report(args.config)
