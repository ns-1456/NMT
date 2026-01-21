"""Base tokenizer class"""

from abc import ABC, abstractmethod
from typing import List, Union


class BaseTokenizer(ABC):
    """Base class for tokenizers."""
    
    @abstractmethod
    def train(self, files: List[str], vocab_size: int) -> None:
        """Train the tokenizer on files.
        
        Args:
            files: List of file paths to train on
            vocab_size: Target vocabulary size
        """
        pass
    
    @abstractmethod
    def encode(self, text: str, **kwargs) -> Union[List[int], dict]:
        """Encode text to token IDs.
        
        Args:
            text: Input text
            **kwargs: Additional encoding parameters
            
        Returns:
            Token IDs or dictionary with token IDs and attention mask
        """
        pass
    
    @abstractmethod
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode token IDs to text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens
            
        Returns:
            Decoded text
        """
        pass
    
    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        pass
    
    @property
    @abstractmethod
    def pad_token_id(self) -> int:
        """Get padding token ID."""
        pass
    
    @property
    @abstractmethod
    def unk_token_id(self) -> int:
        """Get unknown token ID."""
        pass
    
    @property
    @abstractmethod
    def bos_token_id(self) -> int:
        """Get beginning-of-sequence token ID."""
        pass
    
    @property
    @abstractmethod
    def eos_token_id(self) -> int:
        """Get end-of-sequence token ID."""
        pass
