#!/usr/bin/env python3
"""Generate translation examples"""

import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.models.transformer import TransformerNMT
from src.tokenization.bpe import BPETokenizer
from src.tokenization.unigram import UnigramTokenizer


def translate_text(model, source_tokenizer, target_tokenizer, text, device, max_length=128):
    """Translate a single text."""
    model.eval()
    
    # Encode source
    enc_result = source_tokenizer.encode(
        text,
        add_special_tokens=True,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    if isinstance(enc_result, dict):
        input_ids = enc_result['input_ids'].to(device)
    else:
        input_ids = enc_result.to(device)
    
    # Encode
    memory = model.encode(input_ids)
    
    # Greedy decoding
    decoder_input = torch.full(
        (1, 1),
        target_tokenizer.bos_token_id,
        dtype=torch.long,
        device=device
    )
    
    for _ in range(max_length):
        decoder_output = model.decode(decoder_input, memory)
        logits = model.output_projection(decoder_output)
        logits = logits.transpose(0, 1)
        
        next_token_logits = logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        
        decoder_input = torch.cat([decoder_input, next_token], dim=1)
        
        if (next_token == target_tokenizer.eos_token_id).all():
            break
    
    # Decode
    seq = decoder_input[0].tolist()
    if target_tokenizer.eos_token_id in seq:
        seq = seq[:seq.index(target_tokenizer.eos_token_id)]
    if seq and seq[0] == target_tokenizer.bos_token_id:
        seq = seq[1:]
    
    translation = target_tokenizer.decode(seq, skip_special_tokens=True)
    return translation


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate translation examples")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--examples", type=str, nargs="+", default=None,
                       help="Example sentences to translate")
    parser.add_argument("--num-examples", type=int, default=10,
                       help="Number of examples from test set if --examples not provided")
    args = parser.parse_args()
    
    config = load_config(args.config)
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
    
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_config = checkpoint.get('config', {}).get('model', config['model'])
    if 'student' in model_config:
        model_config = model_config['student']
    elif 'teacher' in model_config:
        model_config = model_config['teacher']
    
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
    model = model.to(device)
    model.eval()
    
    # Get examples
    if args.examples:
        examples = args.examples
        references = None
    else:
        # Load from test set
        test_source = splits_dir / "test.source"
        test_target = splits_dir / "test.target"
        
        with open(test_source, 'r', encoding='utf-8') as f:
            examples = [line.strip() for line in f if line.strip()][:args.num_examples]
        
        with open(test_target, 'r', encoding='utf-8') as f:
            references = [line.strip() for line in f if line.strip()][:args.num_examples]
    
    # Translate
    print("\n" + "="*80)
    print("TRANSLATION EXAMPLES")
    print("="*80)
    
    output_file = Path(config['paths']['checkpoint_dir']) / "translation_examples.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, example in enumerate(examples):
            translation = translate_text(model, source_tokenizer, target_tokenizer, example, device)
            
            print(f"\nExample {i+1}:")
            print(f"  English: {example}")
            print(f"  Gujarati: {translation}")
            if references:
                print(f"  Reference: {references[i]}")
            
            f.write(f"Example {i+1}:\n")
            f.write(f"English: {example}\n")
            f.write(f"Gujarati: {translation}\n")
            if references:
                f.write(f"Reference: {references[i]}\n")
            f.write("\n")
    
    print(f"\nExamples saved to: {output_file}")


if __name__ == "__main__":
    main()
