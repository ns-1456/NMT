"""Transformer architecture for NMT"""

import torch
import torch.nn as nn
from typing import Optional
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout rate
        """
        super().__init__()
        # TODO: Implement positional encoding
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply positional encoding.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        # TODO: Implement forward pass
        x = x + self.pe[:, :x.size(0), :].transpose(0,1)

        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """Transformer encoder layer."""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ):
        """Initialize encoder layer.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
        """
        super().__init__()
        # TODO: Implement encoder layer components

        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=False
        )

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def forward(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            src: Source tensor (seq_len, batch_size, d_model)
            src_mask: Attention mask
            src_key_padding_mask: Key padding mask
            
        Returns:
            Encoded tensor
        """
        # TODO: Implement forward pass
        src2 = self.self_attn(
            src, src, src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]

        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(torch.relu(self.linear1(src))))

        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src

class TransformerDecoderLayer(nn.Module):
    """Transformer decoder layer."""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1
    ):
        """Initialize decoder layer.
        
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
        """
        super().__init__()
        # Masked self-attention (for causal masking)
        self.self_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=False
        )
        
        # Cross-attention (query from decoder, key/value from encoder)
        self.multihead_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=False
        )
        
        # Feedforward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization (three norms for three sub-layers)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout layers
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            tgt: Target tensor (seq_len, batch_size, d_model)
            memory: Memory tensor from encoder (seq_len, batch_size, d_model)
            tgt_mask: Target attention mask (causal mask)
            memory_mask: Memory attention mask
            tgt_key_padding_mask: Target key padding mask
            memory_key_padding_mask: Memory key padding mask
            
        Returns:
            Decoded tensor
        """
        # 1. Masked self-attention with residual connection and layer norm
        tgt2 = self.self_attn(
            tgt, tgt, tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )[0]
        
        # Residual connection + dropout + layer norm
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # 2. Cross-attention: query from decoder, key/value from encoder
        tgt2 = self.multihead_attn(
            tgt, memory, memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )[0]
        
        # Residual connection + dropout + layer norm
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        # 3. Feedforward network with residual connection and layer norm
        tgt2 = self.linear2(self.dropout(torch.relu(self.linear1(tgt))))
        
        # Residual connection + dropout + layer norm
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt


class TransformerNMT(nn.Module):
    """Transformer model for Neural Machine Translation.
    
    Supports both small (<50M params) student model and larger teacher model.
    """
    
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 2048,
        max_seq_length: int = 128,
        dropout: float = 0.1,
        pad_token_id: int = 0
    ):
        """Initialize Transformer NMT model.
        
        Args:
            src_vocab_size: Source vocabulary size
            tgt_vocab_size: Target vocabulary size
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Feedforward dimension
            max_seq_length: Maximum sequence length
            dropout: Dropout rate
            pad_token_id: Padding token ID
        """
        super().__init__()
        
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        
        # Embeddings
        self.src_embedding = nn.Embedding(src_vocab_size, d_model, padding_idx=pad_token_id)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model, padding_idx=pad_token_id)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        self.pos_decoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder.
        
        Args:
            sz: Size of the mask
            
        Returns:
            Causal mask
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def create_padding_mask(self, seq: torch.Tensor) -> torch.Tensor:
        """Create padding mask.
        
        Args:
            seq: Input sequence tensor (batch_size, seq_len)
            
        Returns:
            Padding mask (batch_size, seq_len) where True indicates padding
        """
        return (seq == self.pad_token_id)
    
    def encode(
        self,
        src: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Encode source sequence.
        
        Args:
            src: Source tensor (batch_size, seq_len)
            src_key_padding_mask: Key padding mask
            
        Returns:
            Encoded tensor (seq_len, batch_size, d_model)
        """
        # Embedding
        src = src.transpose(0, 1)  # (seq_len, batch_size)
        src_emb = self.src_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        
        # Encoder layers
        for layer in self.encoder_layers:
            src_emb = layer(src_emb, src_key_padding_mask=src_key_padding_mask)
        
        return src_emb
    
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Decode target sequence.
        
        Args:
            tgt: Target tensor (batch_size, seq_len)
            memory: Encoder output (seq_len, batch_size, d_model)
            tgt_mask: Target attention mask
            tgt_key_padding_mask: Target key padding mask
            memory_key_padding_mask: Memory key padding mask
            
        Returns:
            Decoded tensor (seq_len, batch_size, d_model)
        """
        # Embedding
        tgt = tgt.transpose(0, 1)  # (seq_len, batch_size)
        tgt_emb = self.tgt_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_decoder(tgt_emb)
        
        # Decoder layers
        for layer in self.decoder_layers:
            tgt_emb = layer(
                tgt_emb, memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask
            )
        
        return tgt_emb
    
    def forward(
        self,
        input_ids: torch.Tensor,
        decoder_input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> dict:
        """Forward pass.
        
        Args:
            input_ids: Source token IDs (batch_size, src_seq_len)
            decoder_input_ids: Decoder input token IDs (batch_size, tgt_seq_len)
            labels: Target labels (batch_size, tgt_seq_len)
            
        Returns:
            Dictionary with logits and optionally loss
        """
        batch_size = input_ids.size(0)
        src_len = input_ids.size(1)
        tgt_len = decoder_input_ids.size(1)
        
        # Create masks
        src_padding_mask = self.create_padding_mask(input_ids)
        tgt_padding_mask = self.create_padding_mask(decoder_input_ids)
        
        # Causal mask for decoder
        tgt_mask = self.generate_square_subsequent_mask(tgt_len).to(input_ids.device)
        
        # Encode
        memory = self.encode(input_ids, src_key_padding_mask=src_padding_mask)
        
        # Decode
        decoder_output = self.decode(
            decoder_input_ids,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_padding_mask,
            memory_key_padding_mask=src_padding_mask
        )
        
        # Output projection
        logits = self.output_projection(decoder_output)
        logits = logits.transpose(0, 1)  # (batch_size, seq_len, vocab_size)
        
        output = {"logits": logits}
        
        # Compute loss if labels provided
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
            output["loss"] = loss
        
        return output
    
    def count_parameters(self) -> int:
        """Count total number of parameters.
        
        Returns:
            Total parameter count
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_size_mb(self) -> float:
        """Get model size in megabytes.
        
        Returns:
            Model size in MB
        """
        param_size = sum(p.numel() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)


def create_student_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    d_model: int = 512,
    nhead: int = 8,
    num_layers: int = 4,
    dim_feedforward: int = 2048,
    max_seq_length: int = 128,
    dropout: float = 0.1,
    pad_token_id: int = 0
) -> TransformerNMT:
    """Create a small student model (<50M parameters).
    
    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of encoder/decoder layers
        dim_feedforward: Feedforward dimension
        max_seq_length: Maximum sequence length
        dropout: Dropout rate
        pad_token_id: Padding token ID
        
    Returns:
        TransformerNMT model
    """
    model = TransformerNMT(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_seq_length=max_seq_length,
        dropout=dropout,
        pad_token_id=pad_token_id
    )
    
    param_count = model.count_parameters()
    print(f"Student model created with {param_count:,} parameters ({param_count / 1e6:.2f}M)")
    
    if param_count > 50_000_000:
        print(f"WARNING: Model has {param_count / 1e6:.2f}M parameters, exceeding 50M limit!")
    
    return model


def create_teacher_model(
    src_vocab_size: int,
    tgt_vocab_size: int,
    d_model: int = 512,
    nhead: int = 8,
    num_layers: int = 6,
    dim_feedforward: int = 2048,
    max_seq_length: int = 128,
    dropout: float = 0.1,
    pad_token_id: int = 0
) -> TransformerNMT:
    """Create a larger teacher model for knowledge distillation.
    
    Args:
        src_vocab_size: Source vocabulary size
        tgt_vocab_size: Target vocabulary size
        d_model: Model dimension
        nhead: Number of attention heads
        num_layers: Number of encoder/decoder layers
        dim_feedforward: Feedforward dimension
        max_seq_length: Maximum sequence length
        dropout: Dropout rate
        pad_token_id: Padding token ID
        
    Returns:
        TransformerNMT model
    """
    model = TransformerNMT(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers,
        dim_feedforward=dim_feedforward,
        max_seq_length=max_seq_length,
        dropout=dropout,
        pad_token_id=pad_token_id
    )
    
    param_count = model.count_parameters()
    print(f"Teacher model created with {param_count:,} parameters ({param_count / 1e6:.2f}M)")
    
    return model
