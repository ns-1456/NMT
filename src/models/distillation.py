"""Knowledge Distillation for NMT"""

import torch
import torch.nn as nn
from typing import Optional


class DistillationLoss(nn.Module):
    """Knowledge Distillation loss combining hard and soft targets.
    
    Loss = α * hard_loss + (1 - α) * soft_loss
    where:
    - hard_loss: Cross-entropy with ground truth labels
    - soft_loss: KL divergence between student and teacher logits
    """
    
    def __init__(
        self,
        temperature: float = 4.0,
        alpha: float = 0.5,
        ignore_index: int = -100
    ):
        """Initialize distillation loss.
        
        Args:
            temperature: Temperature for softmax (higher = softer distribution)
            alpha: Weight for hard loss (1-alpha for soft loss)
            ignore_index: Index to ignore in loss computation
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.ignore_index = ignore_index
        
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')
        
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: torch.Tensor
    ) -> dict:
        """Compute distillation loss.
        
        Args:
            student_logits: Student model logits (batch_size, seq_len, vocab_size)
            teacher_logits: Teacher model logits (batch_size, seq_len, vocab_size)
            labels: Ground truth labels (batch_size, seq_len)
            
        Returns:
            Dictionary with total loss, hard loss, and soft loss
        """
        batch_size, seq_len, vocab_size = student_logits.shape
        
        # Reshape for loss computation
        student_logits_flat = student_logits.reshape(-1, vocab_size)
        teacher_logits_flat = teacher_logits.reshape(-1, vocab_size)
        labels_flat = labels.reshape(-1)
        
        # Hard loss: Cross-entropy with ground truth
        hard_loss = self.ce_loss(student_logits_flat, labels_flat)
        
        # Soft loss: KL divergence between student and teacher
        # Apply temperature scaling
        student_probs = torch.nn.functional.log_softmax(student_logits_flat / self.temperature, dim=-1)
        teacher_probs = torch.nn.functional.softmax(teacher_logits_flat / self.temperature, dim=-1)
        
        # Mask out padding tokens for soft loss
        mask = (labels_flat != self.ignore_index).float()
        if mask.sum() > 0:
            # Only compute KL loss on non-padding tokens
            student_probs_masked = student_probs * mask.unsqueeze(-1)
            teacher_probs_masked = teacher_probs * mask.unsqueeze(-1)
            
            # Normalize by number of non-padding tokens
            soft_loss = self.kl_loss(student_probs_masked, teacher_probs_masked) * (self.temperature ** 2)
            soft_loss = soft_loss * (mask.sum() / mask.numel())  # Normalize by valid tokens
        else:
            soft_loss = torch.tensor(0.0, device=student_logits.device)
        
        # Combined loss
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        return {
            "loss": total_loss,
            "hard_loss": hard_loss,
            "soft_loss": soft_loss
        }


def compute_distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.5,
    ignore_index: int = -100
) -> dict:
    """Compute distillation loss (functional version).
    
    Args:
        student_logits: Student model logits
        teacher_logits: Teacher model logits
        labels: Ground truth labels
        temperature: Temperature for softmax
        alpha: Weight for hard loss
        ignore_index: Index to ignore
        
    Returns:
        Dictionary with loss components
    """
    loss_fn = DistillationLoss(temperature=temperature, alpha=alpha, ignore_index=ignore_index)
    return loss_fn(student_logits, teacher_logits, labels)
