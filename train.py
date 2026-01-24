#!/usr/bin/env python3
"""
Train T5-Nano (random init) on the processed XLCoST dataset (Python -> C++).

Requirements implemented (per prompt):
- Import model from model_config.py and dataset from data_prep.py
- preprocess_function:
  - prefixes inputs with 'translate Python to C++: '
  - tokenizes inputs (max_length=256) and targets (max_length=512)
  - padding handled via DataCollatorForSeq2Seq
- Use Hugging Face Seq2SeqTrainer with specified Seq2SeqTrainingArguments
- Save final model to ./final_model
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from transformers import (
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

import data_prep
import model_config


TASK_PREFIX = "translate Python to C++: "


def preprocess_function(examples, tokenizer, max_source_len: int, max_target_len: int):
    sources = examples["source"]
    targets = examples["target"]

    # Ensure prefix is present (avoid double-prefixing if data_prep already applied it).
    sources = [
        s if isinstance(s, str) and s.startswith(TASK_PREFIX) else f"{TASK_PREFIX}{s}"
        for s in sources
    ]

    model_inputs = tokenizer(
        sources,
        max_length=max_source_len,
        truncation=True,
    )

    labels = tokenizer(
        text_target=targets,
        max_length=max_target_len,
        truncation=True,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_dir", type=Path, default=Path("t5_nano_checkpoints"))
    ap.add_argument("--final_model_dir", type=Path, default=Path("final_model"))
    ap.add_argument("--max_source_len", type=int, default=256)
    ap.add_argument("--max_target_len", type=int, default=512)
    ap.add_argument("--per_device_batch_size", type=int, default=32)
    ap.add_argument("--num_train_epochs", type=int, default=30)
    args = ap.parse_args()

    tokenizer = model_config.load_tokenizer()
    model = model_config.build_t5_nano(tokenizer)

    # Dataset must exist from data_prep.py (requires `datasets` installed)
    ds = data_prep.load_processed_dataset()

    tokenized = ds.map(
        lambda batch: preprocess_function(
            batch,
            tokenizer=tokenizer,
            max_source_len=args.max_source_len,
            max_target_len=args.max_target_len,
        ),
        batched=True,
        remove_columns=ds["train"].column_names,
        desc="Tokenizing",
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(args.output_dir),
        learning_rate=1e-3,
        per_device_train_batch_size=int(args.per_device_batch_size),
        per_device_eval_batch_size=int(args.per_device_batch_size),
        num_train_epochs=float(args.num_train_epochs),
        weight_decay=0.01,
        warmup_steps=2000,
        fp16=True,
        save_total_limit=2,
        # Reasonable defaults (not specified, but helps Trainer run)
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=False,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized.get("validation"),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()

    args.final_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(args.final_model_dir))
    tokenizer.save_pretrained(str(args.final_model_dir))

    print(f"Saved final model to {args.final_model_dir}")
    return 0


if __name__ == "__main__":
    # Ensure we fail early if no GPU and fp16 is enabled.
    if not torch.cuda.is_available():
        print(
            "WARNING: CUDA is not available. fp16 training will not work on CPU; "
            "set fp16=False in train.py if you need CPU training."
        )
    raise SystemExit(main())

