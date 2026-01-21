"""Evaluation utilities for NMT"""

import torch
from typing import List
from tqdm import tqdm


def compute_bleu(
    predictions: List[str],
    references: List[str],
    tokenize: str = "13a"
) -> float:
    """Compute BLEU score.
    
    Args:
        predictions: List of predicted sentences
        references: List of reference sentences
        tokenize: Tokenization method for BLEU
        
    Returns:
        BLEU score
    """
    from sacrebleu import BLEU
    bleu = BLEU(tokenize=tokenize)
    score = bleu.corpus_score(predictions, [references])
    return score.score


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    tokenizer,
    device: torch.device,
    max_length: int = 128,
    beam_size: int = 5,
    length_penalty: float = 0.6
) -> dict:
    """Evaluate model on a dataset.
    
    Args:
        model: Model to evaluate
        dataloader: DataLoader for evaluation data
        tokenizer: Target tokenizer
        device: Device to run on
        max_length: Maximum generation length
        beam_size: Beam size for beam search (not implemented, using greedy)
        length_penalty: Length penalty (not used in greedy)
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_predictions = []
    all_references = []
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                decoder_input_ids=decoder_input_ids,
                labels=labels
            )
            
            if "loss" in outputs:
                total_loss += outputs["loss"].item()
                num_batches += 1
            
            # Generate predictions (greedy decoding)
            predictions = generate_greedy(
                model, input_ids, tokenizer, device, max_length=max_length
            )
            
            # Decode references
            references = []
            for label_seq in labels:
                # Remove -100 (ignored tokens)
                label_seq = label_seq[label_seq != -100]
                ref = tokenizer.decode(label_seq.tolist(), skip_special_tokens=True)
                references.append(ref)
            
            all_predictions.extend(predictions)
            all_references.extend(references)
    
    # Compute BLEU
    bleu_score = compute_bleu(all_predictions, all_references)
    
    # Average loss
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return {
        "bleu": bleu_score,
        "loss": avg_loss,
        "predictions": all_predictions,
        "references": all_references
    }


def generate_greedy(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    tokenizer,
    device: torch.device,
    max_length: int = 128,
    bos_token_id: int = 2,
    eos_token_id: int = 3,
    pad_token_id: int = 0
) -> List[str]:
    """Generate translations using greedy decoding.
    
    Args:
        model: Model to use for generation
        input_ids: Source token IDs (batch_size, src_seq_len)
        tokenizer: Target tokenizer
        device: Device to run on
        max_length: Maximum generation length
        bos_token_id: Beginning-of-sequence token ID
        eos_token_id: End-of-sequence token ID
        pad_token_id: Padding token ID
        
    Returns:
        List of generated sentences
    """
    model.eval()
    batch_size = input_ids.size(0)
    
    # Encode source
    memory = model.encode(input_ids)
    
    # Initialize decoder input with BOS token
    decoder_input = torch.full(
        (batch_size, 1),
        bos_token_id,
        dtype=torch.long,
        device=device
    )
    
    generated_sequences = []
    
    for _ in range(max_length):
        # Decode
        decoder_output = model.decode(decoder_input, memory)
        
        # Get logits
        logits = model.output_projection(decoder_output)
        logits = logits.transpose(0, 1)  # (batch_size, seq_len, vocab_size)
        
        # Get next token (greedy)
        next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)  # (batch_size, 1)
        
        # Append to decoder input
        decoder_input = torch.cat([decoder_input, next_token], dim=1)
        
        # Check if all sequences have EOS
        if (next_token == eos_token_id).all():
            break
    
    # Decode sequences
    for seq in decoder_input:
        # Remove BOS and everything after EOS
        seq_list = seq.tolist()
        if eos_token_id in seq_list:
            seq_list = seq_list[:seq_list.index(eos_token_id)]
        # Remove BOS
        if seq_list and seq_list[0] == bos_token_id:
            seq_list = seq_list[1:]
        
        generated = tokenizer.decode(seq_list, skip_special_tokens=True)
        generated_sequences.append(generated)
    
    return generated_sequences


def compute_perplexity(loss: float) -> float:
    """Compute perplexity from loss.
    
    Args:
        loss: Cross-entropy loss
        
    Returns:
        Perplexity
    """
    return torch.exp(torch.tensor(loss)).item()
