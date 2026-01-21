#!/usr/bin/env python3
"""Main training script for English-Gujarati NMT"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import argparse
from torch.optim import AdamW

from src.utils.config import load_config
from src.data.dataset import ParallelDataset, get_dataloader
from src.tokenization.bpe import BPETokenizer
from src.tokenization.unigram import UnigramTokenizer
from src.models.transformer import create_student_model, create_teacher_model
from src.training.trainer import NMTTrainer


def load_tokenizers(config, splits_dir):
    """Load or create tokenizers."""
    tokenization_config = config['tokenization']
    tokenizer_type = tokenization_config['type']
    vocab_size = tokenization_config['vocab_size']
    
    # Source tokenizer
    source_tokenizer_path = splits_dir / f"source_tokenizer_{tokenizer_type}.json"
    if tokenizer_type == "bpe":
        source_tokenizer = BPETokenizer(vocab_size=vocab_size)
        if source_tokenizer_path.exists():
            print(f"Loading source tokenizer from {source_tokenizer_path}")
            source_tokenizer.load(source_tokenizer_path)
        else:
            # Train tokenizer
            train_source_file = splits_dir / "train.source"
            print(f"Training source tokenizer on {train_source_file}")
            source_tokenizer.train([str(train_source_file)], vocab_size=vocab_size)
            source_tokenizer.save(source_tokenizer_path)
    else:  # unigram
        source_tokenizer_path = splits_dir / f"source_tokenizer_{tokenizer_type}.model"
        source_tokenizer = UnigramTokenizer(vocab_size=vocab_size)
        if source_tokenizer_path.exists():
            print(f"Loading source tokenizer from {source_tokenizer_path}")
            source_tokenizer.load(source_tokenizer_path)
        else:
            train_source_file = splits_dir / "train.source"
            print(f"Training source tokenizer on {train_source_file}")
            source_tokenizer.train([str(train_source_file)], vocab_size=vocab_size)
            source_tokenizer.save(source_tokenizer_path)
    
    # Target tokenizer
    target_tokenizer_path = splits_dir / f"target_tokenizer_{tokenizer_type}.json"
    if tokenizer_type == "bpe":
        target_tokenizer = BPETokenizer(vocab_size=vocab_size)
        if target_tokenizer_path.exists():
            print(f"Loading target tokenizer from {target_tokenizer_path}")
            target_tokenizer.load(target_tokenizer_path)
        else:
            train_target_file = splits_dir / "train.target"
            print(f"Training target tokenizer on {train_target_file}")
            target_tokenizer.train([str(train_target_file)], vocab_size=vocab_size)
            target_tokenizer.save(target_tokenizer_path)
    else:  # unigram
        target_tokenizer_path = splits_dir / f"target_tokenizer_{tokenizer_type}.model"
        target_tokenizer = UnigramTokenizer(vocab_size=vocab_size)
        if target_tokenizer_path.exists():
            print(f"Loading target tokenizer from {target_tokenizer_path}")
            target_tokenizer.load(target_tokenizer_path)
        else:
            train_target_file = splits_dir / "train.target"
            print(f"Training target tokenizer on {train_target_file}")
            target_tokenizer.train([str(train_target_file)], vocab_size=vocab_size)
            target_tokenizer.save(target_tokenizer_path)
    
    return source_tokenizer, target_tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train English-Gujarati NMT model")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--model-type", type=str, choices=["student", "teacher"], default="student",
                       help="Type of model to train")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Device
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # Data paths
    splits_dir = Path(config['paths']['splits_dir'])
    
    # Check if splits exist, if not create them
    train_source = splits_dir / "train.source"
    if not train_source.exists():
        print("Data splits not found. Please run data preparation first.")
        print("You can use the data download and preprocessing scripts.")
        return
    
    # Load tokenizers
    print("Loading tokenizers...")
    source_tokenizer, target_tokenizer = load_tokenizers(config, splits_dir)
    src_vocab_size = source_tokenizer.get_vocab_size() if hasattr(source_tokenizer, 'get_vocab_size') else (source_tokenizer.vocab_size if hasattr(source_tokenizer, 'vocab_size') else 16000)
    tgt_vocab_size = target_tokenizer.get_vocab_size() if hasattr(target_tokenizer, 'get_vocab_size') else (target_tokenizer.vocab_size if hasattr(target_tokenizer, 'vocab_size') else 16000)
    print(f"Source vocab size: {src_vocab_size}")
    print(f"Target vocab size: {tgt_vocab_size}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = ParallelDataset(
        train_source,
        splits_dir / "train.target",
        source_tokenizer,
        target_tokenizer,
        max_length=config['dataset']['max_length']
    )
    
    val_dataset = ParallelDataset(
        splits_dir / "val.source",
        splits_dir / "val.target",
        source_tokenizer,
        target_tokenizer,
        max_length=config['dataset']['max_length']
    )
    
    # Create dataloaders
    train_dataloader = get_dataloader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True
    )
    
    val_dataloader = get_dataloader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False
    )
    
    # Create model
    print(f"Creating {args.model_type} model...")
    if args.model_type == "student":
        model = create_student_model(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=config['model']['student']['d_model'],
            nhead=config['model']['student']['num_heads'],
            num_layers=config['model']['student']['num_layers'],
            dim_feedforward=config['model']['student']['d_ff'],
            max_seq_length=config['model']['student']['max_seq_length'],
            dropout=config['model']['student']['dropout'],
            pad_token_id=source_tokenizer.pad_token_id
        )
    else:  # teacher
        model = create_teacher_model(
            src_vocab_size=src_vocab_size,
            tgt_vocab_size=tgt_vocab_size,
            d_model=config['model']['teacher']['d_model'],
            nhead=config['model']['teacher']['num_heads'],
            num_layers=config['model']['teacher']['num_layers'],
            dim_feedforward=config['model']['teacher']['d_ff'],
            max_seq_length=config['model']['teacher']['max_seq_length'],
            dropout=config['model']['teacher']['dropout'],
            pad_token_id=source_tokenizer.pad_token_id
        )
    
    model = model.to(device)
    
    # Load teacher model if doing distillation
    teacher_model = None
    if args.model_type == "student" and config.get('distillation', {}).get('enabled', False):
        teacher_checkpoint = config['distillation'].get('teacher_checkpoint')
        if teacher_checkpoint:
            print(f"Loading teacher model from {teacher_checkpoint}...")
            checkpoint = torch.load(teacher_checkpoint, map_location=device)
            teacher_model = create_teacher_model(
                src_vocab_size=src_vocab_size,
                tgt_vocab_size=tgt_vocab_size,
                d_model=config['model']['teacher']['d_model'],
                nhead=config['model']['teacher']['num_heads'],
                num_layers=config['model']['teacher']['num_layers'],
                dim_feedforward=config['model']['teacher']['d_ff'],
                max_seq_length=config['model']['teacher']['max_seq_length'],
                dropout=config['model']['teacher']['dropout'],
                pad_token_id=source_tokenizer.pad_token_id
            )
            teacher_model.load_state_dict(checkpoint['model_state_dict'])
            teacher_model = teacher_model.to(device)
            teacher_model.eval()
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Trainer
    trainer = NMTTrainer(
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        optimizer=optimizer,
        device=device,
        config=config,
        teacher_model=teacher_model,
        target_tokenizer=target_tokenizer
    )
    
    # Train
    resume_from = Path(args.resume) if args.resume else None
    trainer.train(
        num_epochs=config['training']['num_epochs'],
        resume_from=resume_from
    )
    
    print("Training completed!")


if __name__ == "__main__":
    main()
