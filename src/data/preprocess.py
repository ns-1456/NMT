"""Data preprocessing utilities"""

import re
from pathlib import Path
from typing import List, Tuple
import unicodedata


def normalize_unicode(text: str) -> str:
    """Normalize Unicode characters.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Normalize to NFC form
    text = unicodedata.normalize('NFC', text)
    return text


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text.
    
    Args:
        text: Input text
        
    Returns:
        Text with normalized whitespace
    """
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def clean_text(text: str) -> str:
    """Clean and normalize text.
    
    Args:
        text: Input text
        
    Returns:
        Cleaned text
    """
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    return text


def filter_by_length(
    source_lines: List[str],
    target_lines: List[str],
    min_length: int = 3,
    max_length: int = 128
) -> Tuple[List[str], List[str]]:
    """Filter sentence pairs by length.
    
    Args:
        source_lines: Source language sentences
        target_lines: Target language sentences
        min_length: Minimum sentence length (in words)
        max_length: Maximum sentence length (in words)
        
    Returns:
        Filtered (source_lines, target_lines)
    """
    filtered_source = []
    filtered_target = []
    
    for src, tgt in zip(source_lines, target_lines):
        src_words = len(src.split())
        tgt_words = len(tgt.split())
        
        if min_length <= src_words <= max_length and min_length <= tgt_words <= max_length:
            filtered_source.append(src)
            filtered_target.append(tgt)
    
    return filtered_source, filtered_target


def remove_duplicates(
    source_lines: List[str],
    target_lines: List[str]
) -> Tuple[List[str], List[str]]:
    """Remove duplicate sentence pairs.
    
    Args:
        source_lines: Source language sentences
        target_lines: Target language sentences
        
    Returns:
        Deduplicated (source_lines, target_lines)
    """
    seen = set()
    unique_source = []
    unique_target = []
    
    for src, tgt in zip(source_lines, target_lines):
        pair = (src, tgt)
        if pair not in seen:
            seen.add(pair)
            unique_source.append(src)
            unique_target.append(tgt)
    
    return unique_source, unique_target


def preprocess_parallel_files(
    source_file: Path,
    target_file: Path,
    output_source: Path,
    output_target: Path,
    min_length: int = 3,
    max_length: int = 128,
    remove_dup: bool = True
) -> Tuple[int, int]:
    """Preprocess parallel text files.
    
    Args:
        source_file: Path to source language file
        target_file: Path to target language file
        output_source: Path to save processed source file
        output_target: Path to save processed target file
        min_length: Minimum sentence length
        max_length: Maximum sentence length
        remove_dup: Whether to remove duplicates
        
    Returns:
        Tuple of (original_count, processed_count)
    """
    # Read files
    with open(source_file, 'r', encoding='utf-8') as f:
        source_lines = [line.strip() for line in f if line.strip()]
    
    with open(target_file, 'r', encoding='utf-8') as f:
        target_lines = [line.strip() for line in f if line.strip()]
    
    original_count = len(source_lines)
    
    # Ensure same length
    min_len = min(len(source_lines), len(target_lines))
    source_lines = source_lines[:min_len]
    target_lines = target_lines[:min_len]
    
    # Clean text
    source_lines = [clean_text(line) for line in source_lines]
    target_lines = [clean_text(line) for line in target_lines]
    
    # Remove empty lines
    source_lines, target_lines = zip(*[
        (s, t) for s, t in zip(source_lines, target_lines)
        if s and t
    ])
    source_lines = list(source_lines)
    target_lines = list(target_lines)
    
    # Filter by length
    source_lines, target_lines = filter_by_length(
        source_lines, target_lines, min_length, max_length
    )
    
    # Remove duplicates
    if remove_dup:
        source_lines, target_lines = remove_duplicates(source_lines, target_lines)
    
    processed_count = len(source_lines)
    
    # Write processed files
    output_source.parent.mkdir(parents=True, exist_ok=True)
    output_target.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_source, 'w', encoding='utf-8') as f:
        f.write('\n'.join(source_lines) + '\n')
    
    with open(output_target, 'w', encoding='utf-8') as f:
        f.write('\n'.join(target_lines) + '\n')
    
    return original_count, processed_count
