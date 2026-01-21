"""BPE (Byte Pair Encoding) tokenizer implementation"""

from typing import List, Union, Optional
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import BertProcessing
from tokenizers import normalizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents


class BPETokenizer:
    """BPE tokenizer using the tokenizers library.
    
    This tokenizer implements Byte Pair Encoding for sub-word tokenization.
    """
    
    def __init__(
        self,
        vocab_size: int = 16000,
        unk_token: str = "<unk>",
        pad_token: str = "<pad>",
        bos_token: str = "<s>",
        eos_token: str = "</s>"
    ):
        """Initialize BPE tokenizer.
        
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
        
        self.tokenizer = None
        self._is_trained = False
    
    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: Optional[int] = None,
        min_frequency: int = 2
    ) -> None:
        """Train BPE tokenizer on files.
        
        Args:
            files: Single file path or list of file paths
            vocab_size: Vocabulary size (uses self.vocab_size if None)
            min_frequency: Minimum frequency for tokens
        """
        if vocab_size is None:
            vocab_size = self.vocab_size
        
        # Initialize tokenizer
        self.tokenizer = Tokenizer(BPE(unk_token=self.unk_token))
        
        # Set pre-tokenizer
        self.tokenizer.pre_tokenizer = Whitespace()
        
        # Set normalizer
        self.tokenizer.normalizer = normalizers.Sequence([
            NFD(),
            Lowercase(),
            StripAccents()
        ])
        
        # Set post-processor
        self.tokenizer.post_processor = BertProcessing(
            (self.eos_token, 2),
            (self.bos_token, 1)
        )
        
        # Prepare trainer
        special_tokens = [self.unk_token, self.pad_token, self.bos_token, self.eos_token]
        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            min_frequency=min_frequency
        )
        
        # Convert to list if single file
        if isinstance(files, str):
            files = [files]
        
        # Train
        self.tokenizer.train(files, trainer)
        self._is_trained = True
    
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
        
        encoding = self.tokenizer.encode(text)
        token_ids = encoding.ids
        
        # Handle truncation
        if truncation and max_length and len(token_ids) > max_length:
            token_ids = token_ids[:max_length]
        
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
        
        # Remove padding if present
        if isinstance(token_ids, list) and len(token_ids) > 0:
            # Remove padding tokens
            if skip_special_tokens:
                token_ids = [tid for tid in token_ids if tid != self.pad_token_id]
        
        decoded = self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)
        return decoded
    
    def save(self, path: Union[str, Path]) -> None:
        """Save tokenizer to file.
        
        Args:
            path: Path to save tokenizer
        """
        if not self._is_trained:
            raise ValueError("Tokenizer not trained. Cannot save.")
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(str(path))
    
    def load(self, path: Union[str, Path]) -> None:
        """Load tokenizer from file.
        
        Args:
            path: Path to load tokenizer from
        """
        path = Path(path)
        self.tokenizer = Tokenizer.from_file(str(path))
        self._is_trained = True
    
    def get_vocab_size(self) -> int:
        """Get actual vocabulary size from trained tokenizer."""
        if not self._is_trained:
            return self.vocab_size
        return self.tokenizer.get_vocab_size()
    
    @property
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        if not self._is_trained:
            return 0
        return self.tokenizer.token_to_id(self.pad_token) or 0
    
    @property
    def unk_token_id(self) -> int:
        """Get unknown token ID."""
        if not self._is_trained:
            return 1
        return self.tokenizer.token_to_id(self.unk_token) or 1
    
    @property
    def bos_token_id(self) -> int:
        """Get beginning-of-sequence token ID."""
        if not self._is_trained:
            return 2
        return self.tokenizer.token_to_id(self.bos_token) or 2
    
    @property
    def eos_token_id(self) -> int:
        """Get end-of-sequence token ID."""
        if not self._is_trained:
            return 3
        return self.tokenizer.token_to_id(self.eos_token) or 3
