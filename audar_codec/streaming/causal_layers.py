"""
Causal layers for streaming inference.

Provides drop-in replacements for standard layers that support:
- Causal (left-only) padding for convolutions
- Causal attention with KV-cache
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from torchtune.modules import RotaryPositionalEmbeddings

from .cache import KVCache, ConvCache


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution with streaming support.
    
    Uses left-only padding to ensure output at time t only depends on
    inputs at times <= t. Supports streaming via ConvCache.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Convolution kernel size
        stride: Convolution stride
        dilation: Dilation factor
        groups: Number of groups for grouped convolution
        bias: Whether to include bias
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        
        # Causal padding: (kernel_size - 1) * dilation on left only
        self.causal_padding = (kernel_size - 1) * dilation
        
        # Standard conv with no padding (we handle padding manually)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=0,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal padding (for training/non-streaming).
        
        Args:
            x: Input tensor [batch, channels, time]
            
        Returns:
            Output tensor [batch, out_channels, time // stride]
        """
        # Left-pad only for causal behavior
        x = F.pad(x, (self.causal_padding, 0))
        return self.conv(x)
    
    def forward_streaming(
        self,
        x: torch.Tensor,
        cache: Optional[ConvCache] = None,
    ) -> Tuple[torch.Tensor, ConvCache]:
        """
        Streaming forward pass with cache.
        
        Args:
            x: Input chunk [batch, channels, chunk_time]
            cache: ConvCache with previous samples, or None to initialize
            
        Returns:
            out: Output chunk [batch, out_channels, chunk_time // stride]
            cache: Updated ConvCache
        """
        if cache is None:
            cache = ConvCache(padding=self.causal_padding)
        
        # Prepend cached samples
        x_padded = cache.update(x)
        
        # Apply conv (no additional padding needed)
        out = self.conv(x_padded)
        
        return out, cache


class CausalAttention(nn.Module):
    """
    Causal multi-head attention with KV-cache support.
    
    Uses causal masking during training and KV-cache for streaming inference.
    Compatible with the attention pattern in NeuCodec's bs_roformer5.py.
    
    Args:
        dim: Model dimension
        n_heads: Number of attention heads
        rotary_embed: Rotary position embedding module
        max_cache_length: Maximum KV cache length for streaming
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int,
        rotary_embed: Optional[RotaryPositionalEmbeddings] = None,
        max_cache_length: int = 4096,
    ):
        super().__init__()
        assert dim % n_heads == 0, f"dim {dim} must be divisible by n_heads {n_heads}"
        
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.max_cache_length = max_cache_length
        
        # Rotary embeddings (optional, for compatibility)
        self.rotary_embed = rotary_embed
        
        # QKV projection
        self.c_attn = nn.Linear(dim, 3 * dim, bias=False)
        
        # Output projection
        self.c_proj = nn.Linear(dim, dim, bias=False)
        
        # Check for flash attention
        self.flash = hasattr(F, "scaled_dot_product_attention")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with causal masking (for training).
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            
        Returns:
            Output tensor [batch, seq_len, dim]
        """
        B, T, C = x.shape
        
        # Project to Q, K, V
        q, k, v = rearrange(
            self.c_attn(x), "b t (r h d) -> r b h t d",
            r=3, h=self.n_heads
        )
        
        # Apply rotary embeddings if provided
        if self.rotary_embed is not None:
            q = self.rotary_embed(q)
            k = self.rotary_embed(k)
        
        # Causal attention (is_causal=True)
        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=0,
                is_causal=True,  # Key change from NeuCodec V1
            )
        else:
            # Manual implementation with causal mask
            scale = self.head_dim ** -0.5
            attn = (q @ k.transpose(-2, -1)) * scale
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
            attn = attn.masked_fill(mask, float("-inf"))
            attn = F.softmax(attn, dim=-1)
            y = attn @ v
        
        # Reshape and project output
        y = rearrange(y, "b h t d -> b t (h d)")
        return self.c_proj(y)
    
    def forward_streaming(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, KVCache]:
        """
        Streaming forward pass with KV-cache.
        
        Args:
            x: Input chunk [batch, chunk_len, dim]
            kv_cache: KVCache with previous K, V, or None to initialize
            position_offset: Position offset for rotary embeddings
            
        Returns:
            out: Output chunk [batch, chunk_len, dim]
            kv_cache: Updated KVCache
        """
        if kv_cache is None:
            kv_cache = KVCache(max_length=self.max_cache_length)
        
        B, T_chunk, C = x.shape
        
        # Project to Q, K, V for current chunk
        q, k, v = rearrange(
            self.c_attn(x), "b t (r h d) -> r b h t d",
            r=3, h=self.n_heads
        )
        
        # Apply rotary embeddings with position offset
        if self.rotary_embed is not None:
            # Create position indices with offset
            positions = torch.arange(
                position_offset,
                position_offset + T_chunk,
                device=x.device
            )
            q = self.rotary_embed(q)
            k = self.rotary_embed(k)
        
        # Update cache and get full K, V
        full_k, full_v = kv_cache.update(k, v)
        
        # Attention: Q attends to all cached K, V
        # No causal mask needed since we only have past context
        if self.flash:
            y = F.scaled_dot_product_attention(
                q, full_k, full_v,
                attn_mask=None,
                dropout_p=0,
                is_causal=False,  # Full attention to past (all cached)
            )
        else:
            scale = self.head_dim ** -0.5
            attn = (q @ full_k.transpose(-2, -1)) * scale
            attn = F.softmax(attn, dim=-1)
            y = attn @ full_v
        
        # Reshape and project output
        y = rearrange(y, "b h t d -> b t (h d)")
        out = self.c_proj(y)
        
        return out, kv_cache


class RMSNorm(nn.Module):
    """Root Mean Square Normalization."""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = torch.mean(x ** 2, dim=-1, keepdim=True)
        return x * torch.rsqrt(norm_x + self.eps) * self.weight


class MLP(nn.Module):
    """Feed-forward MLP with SiLU activation."""
    
    def __init__(self, dim: int, expansion_factor: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(dim, expansion_factor * dim, bias=False)
        self.silu = nn.SiLU()
        self.fc2 = nn.Linear(expansion_factor * dim, dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.silu(self.fc1(x)))


class CausalTransformerBlock(nn.Module):
    """
    Transformer block with causal attention and streaming support.
    
    Drop-in replacement for NeuCodec's TransformerBlock that supports
    streaming inference via KV-cache.
    
    Args:
        dim: Model dimension
        n_heads: Number of attention heads
        rotary_embed: Rotary position embedding module
        max_cache_length: Maximum KV cache length
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int,
        rotary_embed: Optional[RotaryPositionalEmbeddings] = None,
        max_cache_length: int = 4096,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        
        # Pre-norm architecture
        self.att_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        
        # Causal attention
        self.att = CausalAttention(
            dim=dim,
            n_heads=n_heads,
            rotary_embed=rotary_embed,
            max_cache_length=max_cache_length,
        )
        
        # MLP
        self.mlp = MLP(dim=dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (for training).
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            
        Returns:
            Output tensor [batch, seq_len, dim]
        """
        x = x + self.att(self.att_norm(x))
        x = x + self.mlp(self.ffn_norm(x))
        return x
    
    def forward_streaming(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, KVCache]:
        """
        Streaming forward pass with KV-cache.
        
        Args:
            x: Input chunk [batch, chunk_len, dim]
            kv_cache: KVCache for this layer
            position_offset: Position offset for rotary embeddings
            
        Returns:
            out: Output chunk [batch, chunk_len, dim]
            kv_cache: Updated KVCache
        """
        # Attention with cache
        att_out, kv_cache = self.att.forward_streaming(
            self.att_norm(x),
            kv_cache=kv_cache,
            position_offset=position_offset,
        )
        x = x + att_out
        
        # MLP (no cache needed)
        x = x + self.mlp(self.ffn_norm(x))
        
        return x, kv_cache
