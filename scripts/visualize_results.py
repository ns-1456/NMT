#!/usr/bin/env python3
"""Visualize training results and comparisons"""

import sys
from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config


def plot_training_curves(history_path, output_dir):
    """Plot training curves from history."""
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = [h['epoch'] for h in history]
    train_losses = [h['train']['loss'] for h in history]
    val_losses = [h['val']['loss'] for h in history]
    val_bleus = [h['val']['bleu'] for h in history]
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    axes[0].plot(epochs, train_losses, label='Train Loss', marker='o')
    axes[0].plot(epochs, val_losses, label='Val Loss', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # BLEU curve
    axes[1].plot(epochs, val_bleus, label='Val BLEU', marker='o', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('BLEU Score')
    axes[1].set_title('Validation BLEU Score')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    output_path = output_dir / "training_curves.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved training curves to {output_path}")
    plt.close()


def plot_tokenizer_comparison(comparison_path, output_dir):
    """Plot tokenizer comparison results."""
    with open(comparison_path, 'r') as f:
        comparison = json.load(f)
    
    vocab_sizes = sorted(set(
        size for tok_type in comparison.values()
        for size in tok_type.keys()
    ))
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # OOV rates
    for tok_type in comparison:
        oov_rates = [comparison[tok_type].get(size, {}).get('oov_rate', 0) for size in vocab_sizes]
        axes[0].plot(vocab_sizes, oov_rates, marker='o', label=tok_type.upper())
    axes[0].set_xlabel('Vocabulary Size')
    axes[0].set_ylabel('OOV Rate (%)')
    axes[0].set_title('OOV Rate Comparison')
    axes[0].legend()
    axes[0].grid(True)
    
    # Rare word coverage
    for tok_type in comparison:
        coverages = [comparison[tok_type].get(size, {}).get('rare_word_coverage', 0) for size in vocab_sizes]
        axes[1].plot(vocab_sizes, coverages, marker='s', label=tok_type.upper())
    axes[1].set_xlabel('Vocabulary Size')
    axes[1].set_ylabel('Rare Word Coverage (%)')
    axes[1].set_title('Rare Word Coverage Comparison')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    output_path = output_dir / "tokenizer_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved tokenizer comparison to {output_path}")
    plt.close()


def plot_model_comparison(report_path, output_dir):
    """Plot model comparison charts."""
    with open(report_path, 'r') as f:
        report = json.load(f)
    
    if 'teacher' not in report['models'] or 'student' not in report['models']:
        print("Need both teacher and student models for comparison")
        return
    
    teacher = report['models']['teacher']
    student = report['models']['student']
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Parameter comparison
    models = ['Teacher', 'Student']
    params = [teacher['parameters_millions'], student['parameters_millions']]
    axes[0].bar(models, params, color=['blue', 'green'])
    axes[0].set_ylabel('Parameters (Millions)')
    axes[0].set_title('Model Size Comparison')
    axes[0].grid(True, axis='y')
    for i, v in enumerate(params):
        axes[0].text(i, v, f'{v:.2f}M', ha='center', va='bottom')
    
    # BLEU comparison
    bleus = [teacher['best_bleu'], student['best_bleu']]
    axes[1].bar(models, bleus, color=['blue', 'green'])
    axes[1].set_ylabel('BLEU Score')
    axes[1].set_title('BLEU Score Comparison')
    axes[1].grid(True, axis='y')
    for i, v in enumerate(bleus):
        axes[1].text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    output_path = output_dir / "model_comparison.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved model comparison to {output_path}")
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize training results")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    checkpoint_dir = Path(config['paths']['checkpoint_dir'])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot training curves
    history_path = checkpoint_dir / "training_history.json"
    if history_path.exists():
        plot_training_curves(history_path, checkpoint_dir)
    
    # Plot tokenizer comparison
    tokenizer_comp_path = checkpoint_dir / "tokenizer_comparison.json"
    if tokenizer_comp_path.exists():
        plot_tokenizer_comparison(tokenizer_comp_path, checkpoint_dir)
    
    # Plot model comparison
    report_path = checkpoint_dir / "evaluation_report.json"
    if report_path.exists():
        plot_model_comparison(report_path, checkpoint_dir)
    
    print("\nVisualizations complete!")


if __name__ == "__main__":
    main()
