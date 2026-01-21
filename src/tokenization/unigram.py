"""Unigram tokenizer implementation using SentencePiece"""

from typing import List, Union, Optional
from pathlib import Path
import sentencepiece as spm
import tempfile
import os


class UnigramTokenizer:
    """Unigram tokenizer using SentencePiece.
    
    This tokenizer implements the Unigram language model for sub-word tokenization.
    """
    
    def __init__(
        self,
        vocab_size: int = 16000,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<s>",
        eos_token: str = "</s>"
    ):
        """Initialize Unigram tokenizer.
        
        Args:
            vocab_size: Target vocabulary size
            unk_token: Unknown token string
            pad_token: Padding token string
            bos_token: Beginning-of-sequence token string
            eos_token: End-of-sequence token string
        """
        self.vocab_size = vocab_size
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        
        self.sp_model = None
        self._is_trained = False
    
    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: Optional[int] = None,
        character_coverage: float = 0.9995,
        model_type: str = "unigram"
    ) -> None:
        """Train Unigram tokenizer on files.
        
        Args:
            files: Single file path or list of file paths
            vocab_size: Vocabulary size (uses self.vocab_size if None)
            character_coverage: Character coverage for the model
            model_type: Model type ("unigram", "bpe", "char", "word")
        """
        if vocab_size is None:
            vocab_size = self.vocab_size
        
        # Convert to list if single file
        if isinstance(files, str):
            files = [files]
        
        # Create temporary model file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.model', delete=False) as f:
            model_file = f.name
        
        try:
            # Prepare input file (SentencePiece needs a single input file)
            # If multiple files, concatenate them
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as input_file:
                input_path = input_file.name
                for file_path in files:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        input_file.write(f.read())
            
            # Train SentencePiece model
            spm.SentencePieceTrainer.train(
                input=input_path,
                model_prefix=model_file.replace('.model', ''),
                vocab_size=vocab_size,
                model_type=model_type,
                character_coverage=character_coverage,
                pad_id=0,
                unk_id=1,
                bos_id=2,
                eos_id=3,
                pad_piece=self.pad_token,
                unk_piece=self.unk_token,
                bos_piece=self.bos_token,
                eos_piece=self.eos_token,
                input_sentence_size=1000000,
                shuffle_input_sentence=True
            )
            
            # Load the trained model
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.load(model_file)
            self._is_trained = True
            
        finally:
            # Clean up temporary files
            if os.path.exists(model_file):
                os.remove(model_file)
            if os.path.exists(input_path):
                os.remove(input_path)
    
    def encode(
        self,
        text: str,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: Union[bool, str] = False,
        truncation: bool = False,
        return_tensors: Optional[str] = None
    ) -> Union[List[int], dict]:
        """Encode text to token IDs.
        
        Args:
            text: Input text
            add_special_tokens: Whether to add special tokens
            max_length: Maximum sequence length
            padding: Padding strategy ("max_length", True, or False)
            truncation: Whether to truncate
            return_tensors: Return format ("pt" for PyTorch, None for list)
            
        Returns:
            Token IDs (list or dict with input_ids and attention_mask)
        """
        if not self._is_trained:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        # Encode text
        token_ids = self.sp_model.encode(text, out_type=int)
        
        # Add special tokens
        if add_special_tokens:
            token_ids = [self.bos_token_id] + token_ids + [self.eos_token_id]
        
        # Handle truncation
        if truncation and max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
            # Ensure EOS token if truncated
            if add_special_tokens and token_ids[-1] != self.eos_token_id:
                token_ids[-1] = self.eos_token_id
        
        # Handle padding
        attention_mask = None
        if padding:
            if max_length is None:
                max_length = len(token_ids)
            
            if len(token_ids) < max_length:
                pad_length = max_length - len(token_ids)
                token_ids = token_ids + [self.pad_token_id] * pad_length
                attention_mask = [1] * (max_length - pad_length) + [0] * pad_length
            else:
                attention_mask = [1] * max_length
        
        # Return format
        if return_tensors == "pt":
            import torch
            result = {"input_ids": torch.tensor([token_ids])}
            if attention_mask is not None:
                result["attention_mask"] = torch.tensor([attention_mask])
            return result
        elif attention_mask is not None:
            return {"input_ids": token_ids, "attention_mask": attention_mask}
        else:
            return token_ids
    
    def decode(
        self,
        token_ids: Union[List[int], dict],
        skip_special_tokens: bool = True
    ) -> str:
        """Decode token IDs to text.
        
        Args:
            token_ids: Token IDs (list or dict with input_ids)
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        if not self._is_trained:
            raise ValueError("Tokenizer not trained. Call train() first.")
        
        if isinstance(token_ids, dict):
            token_ids = token_ids["input_ids"]
        
        # Convert to list if tensor
        if hasattr(token_ids, 'tolist'):
            token_ids = token_ids.tolist()
        
        # Remove special tokens if requested
        if skip_special_tokens and isinstance(token_ids, list):
            token_ids = [
                tid for tid in token_ids
                if tid not in [self.pad_token_id, self.bos_token_id, self.eos_token_id, self.unk_token_id]
            ]
        
        decoded = self.sp_model.decode(token_ids)
        return decoded
    
    def save(self, path: Union[str, Path]) -> None:
        """Save tokenizer to file.
        
        Args:
            path: Path to save tokenizer (should end with .model)
        """
        if not self._is_trained:
            raise ValueError("Tokenizer not trained. Cannot save.")
        
        path = Path(path)
        if not path.suffix == '.model':
            path = path.with_suffix('.model')
        
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save SentencePiece model
        model_proto = self.sp_model.serialized_model_proto()
        with open(path, 'wb') as f:
            f.write(model_proto)
    
    def load(self, path: Union[str, Path]) -> None:
        """Load tokenizer from file.
        
        Args:
            path: Path to load tokenizer from
        """
        path = Path(path)
        if not path.suffix == '.model':
            path = path.with_suffix('.model')
        
        self.sp_model = spm.SentencePieceProcessor()
        self.sp_model.load(str(path))
        self._is_trained = True
    
    def get_vocab_size(self) -> int:
        """Get actual vocabulary size from trained tokenizer."""
        if not self._is_trained:
            return self.vocab_size
        return self.sp_model.get_piece_size()
    
    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        return 0
    
    @property
    def unk_token_id(self) -> int:
        """Get unknown token ID."""
        return 1
    
    @property
    def bos_token_id(self) -> int:
        """Get beginning-of-sequence token ID."""
        return 2
    
    @property
    def eos_token_id(self) -> int:
        """Get end-of-sequence token ID."""
        return 3
