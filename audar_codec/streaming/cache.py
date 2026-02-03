"""
Cache management for streaming inference.

Provides stateful containers for KV-cache (attention) and Conv-cache (causal convolutions).
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
import torch
import torch.nn as nn


@dataclass
class KVCache:
    """
    Key-Value cache for causal attention layers.
    
    Stores past K and V tensors to enable incremental decoding
    without recomputing attention over the entire sequence.
    
    Attributes:
        k: Cached keys [batch, num_heads, seq_len, head_dim]
        v: Cached values [batch, num_heads, seq_len, head_dim]
        max_length: Maximum cache length (for sliding window)
    """
    k: Optional[torch.Tensor] = None
    v: Optional[torch.Tensor] = None
    max_length: int = 4096
    
    def update(
        self, 
        new_k: torch.Tensor, 
        new_v: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Append new K, V to cache and return full K, V for attention.
        
        Args:
            new_k: New keys [batch, num_heads, new_len, head_dim]
            new_v: New values [batch, num_heads, new_len, head_dim]
            
        Returns:
            full_k: Concatenated keys [batch, num_heads, total_len, head_dim]
            full_v: Concatenated values [batch, num_heads, total_len, head_dim]
        """
        if self.k is None:
            self.k = new_k
            self.v = new_v
        else:
            self.k = torch.cat([self.k, new_k], dim=2)
            self.v = torch.cat([self.v, new_v], dim=2)
            
            # Apply sliding window if exceeding max length
            if self.k.shape[2] > self.max_length:
                self.k = self.k[:, :, -self.max_length:]
                self.v = self.v[:, :, -self.max_length:]
        
        return self.k, self.v
    
    def get_seq_length(self) -> int:
        """Return current sequence length in cache."""
        return 0 if self.k is None else self.k.shape[2]
    
    def reset(self):
        """Clear the cache."""
        self.k = None
        self.v = None
    
    def clone(self) -> "KVCache":
        """Create a deep copy of the cache."""
        new_cache = KVCache(max_length=self.max_length)
        if self.k is not None:
            new_cache.k = self.k.clone()
            new_cache.v = self.v.clone()
        return new_cache


@dataclass
class ConvCache:
    """
    Cache for causal convolutions.
    
    Stores past input samples to enable streaming convolution
    without padding artifacts at chunk boundaries.
    
    Attributes:
        buffer: Cached input samples [batch, channels, padding_length]
        padding: Required left-padding for causal convolution
    """
    buffer: Optional[torch.Tensor] = None
    padding: int = 0
    
    def update(self, x: torch.Tensor) -> torch.Tensor:
        """
        Prepend cached samples to input and update cache.
        
        Args:
            x: New input [batch, channels, time]
            
        Returns:
            x_padded: Input with prepended cache [batch, channels, time + padding]
        """
        if self.buffer is None:
            # Initialize buffer with zeros
            self.buffer = torch.zeros(
                x.shape[0], x.shape[1], self.padding,
                device=x.device, dtype=x.dtype
            )
        
        # Prepend cache to input
        x_padded = torch.cat([self.buffer, x], dim=-1)
        
        # Update cache with last `padding` samples from padded input
        self.buffer = x_padded[:, :, -self.padding:].clone()
        
        return x_padded
    
    def reset(self):
        """Clear the cache."""
        self.buffer = None
    
    def clone(self) -> "ConvCache":
        """Create a deep copy of the cache."""
        new_cache = ConvCache(padding=self.padding)
        if self.buffer is not None:
            new_cache.buffer = self.buffer.clone()
        return new_cache


@dataclass
class StreamingState:
    """
    Complete streaming state for Audar-Codec.
    
    Encapsulates all caches needed for streaming inference:
    - Encoder conv caches (for causal convolutions)
    - Semantic encoder cache
    - Decoder KV caches (for transformer layers)
    - ISTFT overlap-add buffer
    
    Attributes:
        encoder_conv_caches: List of ConvCache for encoder conv layers
        semantic_cache: Cache for semantic encoder (if streaming-capable)
        decoder_kv_caches: List of KVCache for decoder transformer layers
        istft_buffer: Overlap-add buffer for streaming ISTFT
        position: Current position offset for rotary embeddings
    """
    encoder_conv_caches: List[ConvCache] = field(default_factory=list)
    semantic_cache: Optional[Dict] = None
    decoder_kv_caches: List[KVCache] = field(default_factory=list)
    istft_buffer: Optional[torch.Tensor] = None
    position: int = 0
    
    @classmethod
    def init_for_model(
        cls,
        num_encoder_conv_layers: int = 5,
        encoder_paddings: List[int] = None,
        num_decoder_layers: int = 12,
        max_kv_length: int = 4096,
    ) -> "StreamingState":
        """
        Initialize streaming state for a model configuration.
        
        Args:
            num_encoder_conv_layers: Number of conv layers in encoder
            encoder_paddings: Padding sizes for each encoder conv layer
            num_decoder_layers: Number of transformer layers in decoder
            max_kv_length: Maximum KV cache length
            
        Returns:
            Initialized StreamingState
        """
        if encoder_paddings is None:
            # Default paddings based on NeuCodec encoder architecture
            # kernel_size=7, dilation=(1,3,9) per block
            encoder_paddings = [6] * num_encoder_conv_layers
        
        return cls(
            encoder_conv_caches=[
                ConvCache(padding=p) for p in encoder_paddings
            ],
            decoder_kv_caches=[
                KVCache(max_length=max_kv_length) for _ in range(num_decoder_layers)
            ],
        )
    
    def reset(self):
        """Reset all caches to initial state."""
        for cache in self.encoder_conv_caches:
            cache.reset()
        if self.semantic_cache is not None:
            self.semantic_cache = None
        for cache in self.decoder_kv_caches:
            cache.reset()
        self.istft_buffer = None
        self.position = 0
    
    def clone(self) -> "StreamingState":
        """Create a deep copy of the entire state."""
        new_state = StreamingState(
            encoder_conv_caches=[c.clone() for c in self.encoder_conv_caches],
            semantic_cache=self.semantic_cache.copy() if self.semantic_cache else None,
            decoder_kv_caches=[c.clone() for c in self.decoder_kv_caches],
            istft_buffer=self.istft_buffer.clone() if self.istft_buffer is not None else None,
            position=self.position,
        )
        return new_state
