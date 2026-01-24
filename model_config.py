#!/usr/bin/env python3
"""
T5-Nano model configuration (from scratch, random init).

Requirements implemented:
- Load tokenizer from ./custom_tokenizer using PreTrainedTokenizerFast
- Define T5Config with Nano hyperparameters
- Initialize T5ForConditionalGeneration(config) (NO from_pretrained)
- Print total parameter count in __main__
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast, T5Config, T5ForConditionalGeneration


TOKENIZER_DIR = Path("custom_tokenizer")


def load_tokenizer(tokenizer_dir: Path = TOKENIZER_DIR) -> PreTrainedTokenizerFast:
    vocab_path = tokenizer_dir / "vocab.json"
    merges_path = tokenizer_dir / "merges.txt"

    if not vocab_path.exists() or not merges_path.exists():
        raise FileNotFoundError(
            f"Expected tokenizer files not found in {tokenizer_dir}. "
            "Run train_tokenizer.py first to create vocab.json and merges.txt."
        )

    backend = ByteLevelBPETokenizer(str(vocab_path), str(merges_path))
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        bos_token="<s>",
        pad_token="<pad>",
        eos_token="</s>",
        unk_token="<unk>",
        mask_token="<mask>",
    )
    return tokenizer


def build_t5_nano(tokenizer: PreTrainedTokenizerFast) -> T5ForConditionalGeneration:
    config = T5Config(
        vocab_size=16_000,
        d_model=256,
        d_kv=32,
        d_ff=1024,
        num_layers=6,
        num_decoder_layers=6,
        num_heads=4,
        dropout_rate=0.2,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        decoder_start_token_id=tokenizer.pad_token_id,
    )

    # Random initialization (do NOT use from_pretrained)
    model = T5ForConditionalGeneration(config)
    return model


def count_parameters(model: T5ForConditionalGeneration) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    tok = load_tokenizer()
    model = build_t5_nano(tok)
    n_params = count_parameters(model)
    print(f"T5-Nano parameter count: {n_params:,}")
    if 20_000_000 <= n_params <= 40_000_000:
        print("OK: parameter count is within the expected 20M–40M range.")
    else:
        print("WARNING: parameter count is outside the expected 20M–40M range.")

