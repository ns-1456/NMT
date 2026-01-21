"""Additional metrics and utilities"""

from typing import List, Dict
import torch


def count_parameters(model: torch.nn.Module) -> int:
    """Count total trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: torch.nn.Module) -> float:
    """Get model size in megabytes.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model size in MB
    """
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 * 1024)


def analyze_tokenizer_coverage(
    tokenizer,
    texts: List[str],
    rare_word_threshold: int = 5
) -> Dict[str, float]:
    """Analyze tokenizer coverage on texts.
    
    Args:
        tokenizer: Tokenizer to analyze
        texts: List of texts to analyze
        rare_word_threshold: Threshold for considering a word rare
        
    Returns:
        Dictionary with coverage metrics
    """
    total_words = 0
    oov_count = 0
    rare_word_count = 0
    rare_word_oov = 0
    
    word_freq = {}
    
    # Count word frequencies
    for text in texts:
        words = text.split()
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1
            total_words += 1
    
    # Analyze coverage
    for text in texts:
        words = text.split()
        for word in words:
            is_rare = word_freq[word] <= rare_word_threshold
            if is_rare:
                rare_word_count += 1
            
            # Check if word can be tokenized (no OOV)
            try:
                tokens = tokenizer.encode(word, add_special_tokens=False)
                if isinstance(tokens, dict):
                    tokens = tokens.get('input_ids', [])
                if not tokens or (len(tokens) == 1 and tokens[0] == tokenizer.unk_token_id):
                    oov_count += 1
                    if is_rare:
                        rare_word_oov += 1
            except:
                oov_count += 1
                if is_rare:
                    rare_word_oov += 1
    
    oov_rate = (oov_count / total_words) * 100 if total_words > 0 else 0.0
    rare_word_coverage = (1 - rare_word_oov / rare_word_count) * 100 if rare_word_count > 0 else 0.0
    
    return {
        'oov_rate': oov_rate,
        'rare_word_coverage': rare_word_coverage,
        'total_words': total_words,
        'oov_words': oov_count,
        'rare_words': rare_word_count,
        'rare_word_oov': rare_word_oov
    }
