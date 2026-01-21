"""Training utilities for NMT"""

import torch
import torch.nn as nn
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import LambdaLR
from pathlib import Path
from typing import Optional, Dict, Any
from tqdm import tqdm
import json
import os

from ..models.distillation import DistillationLoss
from .evaluator import evaluate_model


class NMTTrainer:
    """Trainer for Neural Machine Translation models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        config: Dict[str, Any],
        teacher_model: Optional[nn.Module] = None,
        target_tokenizer = None
    ):
        """Initialize trainer.
        
        Args:
            model: Model to train
            train_dataloader: Training data loader
            val_dataloader: Validation data loader
            optimizer: Optimizer
            device: Device to train on
            config: Training configuration
            teacher_model: Optional teacher model for distillation
            target_tokenizer: Target tokenizer for evaluation
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimizer = optimizer
        self.device = device
        self.config = config
        self.teacher_model = teacher_model
        self.target_tokenizer = target_tokenizer
        
        # Training state
        self.global_step = 0
        self.best_bleu = 0.0
        self.training_history = []
        
        # Distillation
        self.use_distillation = teacher_model is not None and config.get('distillation', {}).get('enabled', False)
        if self.use_distillation:
            self.distillation_loss = DistillationLoss(
                temperature=config['distillation'].get('temperature', 4.0),
                alpha=config['distillation'].get('alpha', 0.5)
            )
            self.teacher_model.eval()
        
        # Learning rate scheduler
        self.scheduler = self._create_scheduler()
        
        # Checkpoint directory
        self.checkpoint_dir = Path(config['paths']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _create_scheduler(self) -> LambdaLR:
        """Create learning rate scheduler with warmup."""
        warmup_steps = self.config['training'].get('warmup_steps', 4000)
        num_training_steps = len(self.train_dataloader) * self.config['training'].get('num_epochs', 50)
        
        def lr_lambda(current_step: int) -> float:
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps)))
        
        return LambdaLR(self.optimizer, lr_lambda)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            epoch: Current epoch number
            
        Returns:
            Dictionary with training metrics
        """
        self.model.train()
        total_loss = 0.0
        total_hard_loss = 0.0
        total_soft_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            decoder_input_ids = batch['decoder_input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            if self.use_distillation:
                # Get teacher predictions
                with torch.no_grad():
                    teacher_outputs = self.teacher_model(
                        input_ids=input_ids,
                        decoder_input_ids=decoder_input_ids
                    )
                    teacher_logits = teacher_outputs['logits']
                
                # Get student predictions
                student_outputs = self.model(
                    input_ids=input_ids,
                    decoder_input_ids=decoder_input_ids
                )
                student_logits = student_outputs['logits']
                
                # Compute distillation loss
                loss_dict = self.distillation_loss(
                    student_logits, teacher_logits, labels
                )
                loss = loss_dict['loss']
                hard_loss = loss_dict['hard_loss'].item()
                soft_loss = loss_dict['soft_loss'].item()
            else:
                # Standard training
                outputs = self.model(
                    input_ids=input_ids,
                    decoder_input_ids=decoder_input_ids,
                    labels=labels
                )
                loss = outputs['loss']
                hard_loss = loss.item()
                soft_loss = 0.0
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            max_grad_norm = self.config['training'].get('max_grad_norm', 1.0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            
            # Optimizer step
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()
            
            # Update metrics
            total_loss += loss.item()
            total_hard_loss += hard_loss
            total_soft_loss += soft_loss
            num_batches += 1
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Logging
            if self.global_step % self.config['training'].get('logging_steps', 100) == 0:
                avg_loss = total_loss / num_batches
                print(f"Step {self.global_step}: Loss = {avg_loss:.4f}")
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        avg_hard_loss = total_hard_loss / num_batches if num_batches > 0 else 0.0
        avg_soft_loss = total_soft_loss / num_batches if num_batches > 0 else 0.0
        
        return {
            'loss': avg_loss,
            'hard_loss': avg_hard_loss,
            'soft_loss': avg_soft_loss
        }
    
    def validate(self) -> Dict[str, float]:
        """Validate model.
        
        Returns:
            Dictionary with validation metrics
        """
        if self.target_tokenizer is None:
            # Simple validation without BLEU
            self.model.eval()
            total_loss = 0.0
            num_batches = 0
            
            with torch.no_grad():
                for batch in tqdm(self.val_dataloader, desc="Validating"):
                    input_ids = batch['input_ids'].to(self.device)
                    decoder_input_ids = batch['decoder_input_ids'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        decoder_input_ids=decoder_input_ids,
                        labels=labels
                    )
                    
                    total_loss += outputs['loss'].item()
                    num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            return {'loss': avg_loss, 'bleu': 0.0}
        else:
            # Full evaluation with BLEU
            eval_results = evaluate_model(
                self.model,
                self.val_dataloader,
                self.target_tokenizer,
                self.device,
                max_length=self.config.get('evaluation', {}).get('max_length', 128)
            )
            return {
                'loss': eval_results['loss'],
                'bleu': eval_results['bleu']
            }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            epoch: Current epoch
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_bleu': self.best_bleu,
            'config': self.config
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            print(f"Saved best model with BLEU: {self.best_bleu:.2f}")
        
        # Keep only last N checkpoints
        save_total_limit = self.config['training'].get('save_total_limit', 3)
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        if len(checkpoints) > save_total_limit:
            for old_checkpoint in checkpoints[:-save_total_limit]:
                old_checkpoint.unlink()
    
    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint.get('global_step', 0)
        self.best_bleu = checkpoint.get('best_bleu', 0.0)
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    def train(self, num_epochs: int, resume_from: Optional[Path] = None) -> None:
        """Train the model.
        
        Args:
            num_epochs: Number of epochs to train
            resume_from: Optional path to checkpoint to resume from
        """
        if resume_from is not None:
            self.load_checkpoint(resume_from)
        
        for epoch in range(1, num_epochs + 1):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Update best BLEU
            is_best = val_metrics['bleu'] > self.best_bleu
            if is_best:
                self.best_bleu = val_metrics['bleu']
            
            # Save checkpoint
            if epoch % self.config['training'].get('save_steps', 5000) == 0 or is_best:
                self.save_checkpoint(epoch, is_best=is_best)
            
            # Log metrics
            print(f"Epoch {epoch}:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}")
            if self.use_distillation:
                print(f"  Train Hard Loss: {train_metrics['hard_loss']:.4f}")
                print(f"  Train Soft Loss: {train_metrics['soft_loss']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}")
            print(f"  Val BLEU: {val_metrics['bleu']:.2f}")
            print(f"  Best BLEU: {self.best_bleu:.2f}")
            
            # Save training history
            self.training_history.append({
                'epoch': epoch,
                'train': train_metrics,
                'val': val_metrics
            })
            
            # Save history to file
            history_path = self.checkpoint_dir / "training_history.json"
            with open(history_path, 'w') as f:
                json.dump(self.training_history, f, indent=2)
