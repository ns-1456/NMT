#!/usr/bin/env python3
"""
Simple inference/demo for the trained Python -> C++ translator.

Loads model + tokenizer from ./final_model and runs beam search generation.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration


TASK_PREFIX = "translate Python to C++: "
FINAL_MODEL_DIR = Path("final_model")


tokenizer = None
model = None
device = None


def _lazy_load():
    global tokenizer, model, device
    if tokenizer is not None and model is not None:
        return

    if not FINAL_MODEL_DIR.exists():
        raise FileNotFoundError(
            f"{FINAL_MODEL_DIR} not found. Train first with: python3 train.py"
        )

    tokenizer = AutoTokenizer.from_pretrained(str(FINAL_MODEL_DIR), use_fast=True)
    model = T5ForConditionalGeneration.from_pretrained(str(FINAL_MODEL_DIR))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()


def translate(python_code_str: str) -> str:
    """
    Translate a Python snippet to C++ and print the decoded result.
    """
    _lazy_load()

    assert tokenizer is not None
    assert model is not None
    assert device is not None

    text = python_code_str
    if not text.startswith(TASK_PREFIX):
        text = TASK_PREFIX + text

    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            num_beams=4,
            max_length=512,
        )

    decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True)
    print(decoded)
    return decoded


if __name__ == "__main__":
    sample_python = """\
def sum_upto(n):
    s = 0
    for i in range(n + 1):
        s += i
    return s
"""
    print("=== Python input ===")
    print(sample_python)
    print("=== Generated C++ ===")
    translate(sample_python)

