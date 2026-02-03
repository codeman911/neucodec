"""
Streaming Codec Decoder for Audar-Codec.

Converts NeuCodec V1 decoder to streaming-capable architecture with:
- Causal convolutions (left-only padding)
- Causal attention with KV-cache
- Streaming ISTFT with overlap-add buffer

Supports 100% weight reuse from NeuCodec V1.
"""

from typing import Optional, Tuple, List
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..streaming.cache import KVCache, ConvCache, StreamingState
from ..streaming.causal_layers import CausalConv1d, RMSNorm
from ..streaming.istft_streaming import StreamingISTFT, StreamingISTFTHead


@dataclass
class DecoderStreamingState:
    """State for streaming decoder inference."""
    # Embed conv cache
    embed_cache: Optional[ConvCache] = None
    # Prior net conv caches (2 blocks x 2 convs each)
    prior_conv_caches: Optional[List[List[ConvCache]]] = None
    # Transformer KV caches (12 layers)
    transformer_kv_caches: Optional[List[KVCache]] = None
    # Post net conv caches (2 blocks x 2 convs each)
    post_conv_caches: Optional[List[List[ConvCache]]] = None
    # ISTFT overlap buffer
    istft_buffer: Optional[torch.Tensor] = None
    # Position offset for RoPE
    position_offset: int = 0
    
    def reset(self):
        """Reset all caches."""
        if self.embed_cache is not None:
            self.embed_cache.reset()
        if self.prior_conv_caches is not None:
            for block_caches in self.prior_conv_caches:
                for cache in block_caches:
                    cache.reset()
        if self.transformer_kv_caches is not None:
            for cache in self.transformer_kv_caches:
                cache.reset()
        if self.post_conv_caches is not None:
            for block_caches in self.post_conv_caches:
                for cache in block_caches:
                    cache.reset()
        self.istft_buffer = None
        self.position_offset = 0


def nonlinearity(x: torch.Tensor) -> torch.Tensor:
    """Swish activation."""
    return x * torch.sigmoid(x)


class CausalGroupNorm(nn.Module):
    """
    GroupNorm that works with causal convolutions.
    Standard GroupNorm - no changes needed as it operates per-timestep.
    """
    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.GroupNorm(num_groups, num_channels, eps=eps, affine=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class CausalResnetBlock(nn.Module):
    """
    Causal ResNet block for streaming inference.
    
    Converts symmetric Conv1d (padding=1) to causal Conv1d (left-only padding).
    Maintains weight compatibility with NeuCodec V1.
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: Optional[int] = None,
        dropout: float = 0.1,
        conv_shortcut: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        
        # Causal convolutions with left-only padding
        self.norm1 = CausalGroupNorm(32, in_channels)
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size=3)
        
        self.norm2 = CausalGroupNorm(32, out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size=3)
        
        # Shortcut connection
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = CausalConv1d(in_channels, out_channels, kernel_size=3)
            else:
                # 1x1 conv doesn't need causal padding
                self.nin_shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full sequence forward (training mode)."""
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        
        return x + h
    
    def forward_streaming(
        self,
        x: torch.Tensor,
        conv1_cache: Optional[ConvCache] = None,
        conv2_cache: Optional[ConvCache] = None,
        shortcut_cache: Optional[ConvCache] = None,
    ) -> Tuple[torch.Tensor, ConvCache, ConvCache, Optional[ConvCache]]:
        """
        Streaming forward with conv caches.
        
        Args:
            x: Input chunk [B, C, T]
            conv1_cache: Cache for first conv
            conv2_cache: Cache for second conv
            shortcut_cache: Cache for shortcut conv (if needed)
            
        Returns:
            output: Output chunk [B, C, T]
            conv1_cache: Updated cache
            conv2_cache: Updated cache
            shortcut_cache: Updated cache (or None)
        """
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h, conv1_cache = self.conv1.forward_streaming(h, conv1_cache)
        
        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h, conv2_cache = self.conv2.forward_streaming(h, conv2_cache)
        
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x, shortcut_cache = self.conv_shortcut.forward_streaming(x, shortcut_cache)
            else:
                x = self.nin_shortcut(x)
        
        return x + h, conv1_cache, conv2_cache, shortcut_cache
    
    @classmethod
    def from_v1_block(cls, v1_block: nn.Module) -> "CausalResnetBlock":
        """
        Create CausalResnetBlock from NeuCodec V1 ResnetBlock.
        
        Copies all weights, converting symmetric convs to causal.
        """
        block = cls(
            in_channels=v1_block.in_channels,
            out_channels=v1_block.out_channels,
            dropout=v1_block.dropout.p if hasattr(v1_block.dropout, 'p') else 0.1,
            conv_shortcut=v1_block.use_conv_shortcut,
        )
        
        # Copy normalization weights
        block.norm1.norm.load_state_dict(v1_block.norm1.state_dict())
        block.norm2.norm.load_state_dict(v1_block.norm2.state_dict())
        
        # Copy conv weights (same weights, different padding strategy)
        block.conv1.conv.weight.data.copy_(v1_block.conv1.weight.data)
        block.conv1.conv.bias.data.copy_(v1_block.conv1.bias.data)
        block.conv2.conv.weight.data.copy_(v1_block.conv2.weight.data)
        block.conv2.conv.bias.data.copy_(v1_block.conv2.bias.data)
        
        # Copy shortcut if present
        if v1_block.in_channels != v1_block.out_channels:
            if v1_block.use_conv_shortcut:
                block.conv_shortcut.conv.weight.data.copy_(v1_block.conv_shortcut.weight.data)
                block.conv_shortcut.conv.bias.data.copy_(v1_block.conv_shortcut.bias.data)
            else:
                block.nin_shortcut.load_state_dict(v1_block.nin_shortcut.state_dict())
        
        return block


class CausalTransformerBlock(nn.Module):
    """
    Causal transformer block with KV-cache support.
    
    Converts NeuCodec V1's bidirectional attention to causal attention
    while maintaining weight compatibility.
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int,
        rope_dim: int = 64,
        max_seq_len: int = 8192,
    ):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.rope_dim = rope_dim
        
        # Pre-norm with RMSNorm
        self.att_norm = RMSNorm(dim)
        self.ffn_norm = RMSNorm(dim)
        
        # Attention projections
        self.c_attn = nn.Linear(dim, 3 * dim, bias=False)
        self.c_proj = nn.Linear(dim, dim, bias=False)
        
        # MLP
        self.fc1 = nn.Linear(dim, 4 * dim, bias=False)
        self.fc2 = nn.Linear(4 * dim, dim, bias=False)
        
        # RoPE frequencies
        self._init_rope(max_seq_len)
    
    def _init_rope(self, max_seq_len: int):
        """Initialize rotary position embedding frequencies."""
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.rope_dim, 2).float() / self.rope_dim))
        self.register_buffer("inv_freq", inv_freq)
        
        # Precompute cos/sin for common sequence lengths
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)
    
    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embeddings to Q and K."""
        seq_len = q.shape[2]
        
        # Get cos/sin for positions
        cos = self.cos_cached[position_offset:position_offset + seq_len]
        sin = self.sin_cached[position_offset:position_offset + seq_len]
        
        # Reshape for broadcasting: [1, 1, seq_len, rope_dim//2]
        cos = cos.unsqueeze(0).unsqueeze(0)
        sin = sin.unsqueeze(0).unsqueeze(0)
        
        # Apply RoPE to first rope_dim dimensions only
        q_rope = q[..., :self.rope_dim]
        k_rope = k[..., :self.rope_dim]
        
        # Split into even/odd
        q1, q2 = q_rope[..., ::2], q_rope[..., 1::2]
        k1, k2 = k_rope[..., ::2], k_rope[..., 1::2]
        
        # Apply rotation
        q_rotated = torch.stack([
            q1 * cos - q2 * sin,
            q2 * cos + q1 * sin,
        ], dim=-1).flatten(-2)
        
        k_rotated = torch.stack([
            k1 * cos - k2 * sin,
            k2 * cos + k1 * sin,
        ], dim=-1).flatten(-2)
        
        # Concatenate with non-rotated dimensions
        if self.rope_dim < self.head_dim:
            q = torch.cat([q_rotated, q[..., self.rope_dim:]], dim=-1)
            k = torch.cat([k_rotated, k[..., self.rope_dim:]], dim=-1)
        else:
            q = q_rotated
            k = k_rotated
        
        return q, k
    
    def forward(
        self,
        x: torch.Tensor,
        position_offset: int = 0,
    ) -> torch.Tensor:
        """
        Full sequence forward (training mode).
        
        Uses causal attention mask for autoregressive behavior.
        """
        B, T, C = x.shape
        
        # Attention with pre-norm
        residual = x
        x = self.att_norm(x)
        
        # QKV projection
        qkv = self.c_attn(x)
        q, k, v = rearrange(qkv, "b t (r h d) -> r b h t d", r=3, h=self.n_heads)
        
        # Apply RoPE
        q, k = self._apply_rope(q, k, position_offset)
        
        # Causal self-attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = rearrange(y, "b h t d -> b t (h d)")
        y = self.c_proj(y)
        
        x = residual + y
        
        # MLP with pre-norm
        residual = x
        x = self.ffn_norm(x)
        x = self.fc1(x)
        x = F.silu(x)
        x = self.fc2(x)
        x = residual + x
        
        return x
    
    def forward_streaming(
        self,
        x: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        position_offset: int = 0,
    ) -> Tuple[torch.Tensor, KVCache, int]:
        """
        Streaming forward with KV-cache.
        
        Args:
            x: Input chunk [B, T_chunk, dim]
            kv_cache: Cached K and V from previous chunks
            position_offset: Position offset for RoPE
            
        Returns:
            output: Output chunk [B, T_chunk, dim]
            kv_cache: Updated KV cache
            new_position_offset: Updated position offset
        """
        B, T, C = x.shape
        
        if kv_cache is None:
            kv_cache = KVCache()
        
        # Attention with pre-norm
        residual = x
        x = self.att_norm(x)
        
        # QKV projection for new tokens
        qkv = self.c_attn(x)
        q, k, v = rearrange(qkv, "b t (r h d) -> r b h t d", r=3, h=self.n_heads)
        
        # Apply RoPE to new tokens
        q, k = self._apply_rope(q, k, position_offset)
        
        # Update cache with new K, V
        k_full, v_full = kv_cache.update(k, v)
        
        # Attention: Q attends to all cached K, V (no causal mask needed - cache handles it)
        y = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=False)
        y = rearrange(y, "b h t d -> b t (h d)")
        y = self.c_proj(y)
        
        x = residual + y
        
        # MLP with pre-norm
        residual = x
        x = self.ffn_norm(x)
        x = self.fc1(x)
        x = F.silu(x)
        x = self.fc2(x)
        x = residual + x
        
        return x, kv_cache, position_offset + T
    
    @classmethod
    def from_v1_block(cls, v1_block: nn.Module, rope_dim: int = 64) -> "CausalTransformerBlock":
        """
        Create CausalTransformerBlock from NeuCodec V1 TransformerBlock.
        
        Copies attention and MLP weights, switching from bidirectional to causal.
        """
        block = cls(
            dim=v1_block.dim,
            n_heads=v1_block.n_heads,
            rope_dim=rope_dim,
        )
        
        # Copy attention weights
        block.c_attn.load_state_dict(v1_block.att.c_attn.state_dict())
        block.c_proj.load_state_dict(v1_block.att.c_proj.state_dict())
        
        # Copy MLP weights
        block.fc1.load_state_dict(v1_block.mlp.fc1.state_dict())
        block.fc2.load_state_dict(v1_block.mlp.fc2.state_dict())
        
        # Copy norm weights
        block.att_norm.weight.data.copy_(v1_block.att_norm.weight.data)
        block.ffn_norm.weight.data.copy_(v1_block.ffn_norm.weight.data)
        
        return block


class CausalVocosBackbone(nn.Module):
    """
    Causal version of VocosBackbone for streaming inference.
    
    Architecture:
    - embed: CausalConv1d (kernel=7)
    - prior_net: 2x CausalResnetBlock
    - transformers: 12x CausalTransformerBlock
    - post_net: 2x CausalResnetBlock
    - final_layer_norm: LayerNorm
    """
    
    def __init__(
        self,
        hidden_dim: int = 1024,
        depth: int = 12,
        heads: int = 16,
        rope_dim: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.heads = heads
        
        # Embedding conv (causal)
        self.embed = CausalConv1d(hidden_dim, hidden_dim, kernel_size=7)
        
        # Prior network (2 ResNet blocks)
        self.prior_net = nn.ModuleList([
            CausalResnetBlock(hidden_dim, hidden_dim, dropout=dropout),
            CausalResnetBlock(hidden_dim, hidden_dim, dropout=dropout),
        ])
        
        # Transformer blocks (causal attention)
        self.transformers = nn.ModuleList([
            CausalTransformerBlock(hidden_dim, heads, rope_dim)
            for _ in range(depth)
        ])
        
        # Post network (2 ResNet blocks)
        self.post_net = nn.ModuleList([
            CausalResnetBlock(hidden_dim, hidden_dim, dropout=dropout),
            CausalResnetBlock(hidden_dim, hidden_dim, dropout=dropout),
        ])
        
        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full sequence forward (training mode).
        
        Args:
            x: Input [B, T, hidden_dim]
            
        Returns:
            output: [B, T, hidden_dim]
        """
        # Embed
        x = x.transpose(1, 2)  # [B, hidden_dim, T]
        x = self.embed(x)
        
        # Prior net
        for block in self.prior_net:
            x = block(x)
        
        # Transformers
        x = x.transpose(1, 2)  # [B, T, hidden_dim]
        for transformer in self.transformers:
            x = transformer(x)
        
        # Post net
        x = x.transpose(1, 2)  # [B, hidden_dim, T]
        for block in self.post_net:
            x = block(x)
        
        # Final norm
        x = x.transpose(1, 2)  # [B, T, hidden_dim]
        x = self.final_layer_norm(x)
        
        return x
    
    def forward_streaming(
        self,
        x: torch.Tensor,
        state: Optional[DecoderStreamingState] = None,
    ) -> Tuple[torch.Tensor, DecoderStreamingState]:
        """
        Streaming forward with state caches.
        
        Args:
            x: Input chunk [B, T_chunk, hidden_dim]
            state: Streaming state from previous call
            
        Returns:
            output: Output chunk [B, T_chunk, hidden_dim]
            state: Updated streaming state
        """
        if state is None:
            state = DecoderStreamingState()
        
        # Initialize caches if needed
        if state.embed_cache is None:
            state.embed_cache = ConvCache(padding=self.embed.causal_padding)
        if state.prior_conv_caches is None:
            state.prior_conv_caches = [
                [ConvCache(padding=2), ConvCache(padding=2)]  # kernel=3 -> padding=2
                for _ in range(2)
            ]
        if state.transformer_kv_caches is None:
            state.transformer_kv_caches = [KVCache() for _ in range(self.depth)]
        if state.post_conv_caches is None:
            state.post_conv_caches = [
                [ConvCache(padding=2), ConvCache(padding=2)]
                for _ in range(2)
            ]
        
        # Embed
        x = x.transpose(1, 2)  # [B, hidden_dim, T]
        x, state.embed_cache = self.embed.forward_streaming(x, state.embed_cache)
        
        # Prior net
        for i, block in enumerate(self.prior_net):
            x, state.prior_conv_caches[i][0], state.prior_conv_caches[i][1], _ = \
                block.forward_streaming(x, state.prior_conv_caches[i][0], state.prior_conv_caches[i][1])
        
        # Transformers
        x = x.transpose(1, 2)  # [B, T, hidden_dim]
        for i, transformer in enumerate(self.transformers):
            x, state.transformer_kv_caches[i], state.position_offset = \
                transformer.forward_streaming(x, state.transformer_kv_caches[i], state.position_offset)
        # Reset position offset after all transformers (they share the same offset)
        state.position_offset = state.transformer_kv_caches[0].get_seq_length() if state.transformer_kv_caches[0].k is not None else 0
        
        # Post net
        x = x.transpose(1, 2)  # [B, hidden_dim, T]
        for i, block in enumerate(self.post_net):
            x, state.post_conv_caches[i][0], state.post_conv_caches[i][1], _ = \
                block.forward_streaming(x, state.post_conv_caches[i][0], state.post_conv_caches[i][1])
        
        # Final norm
        x = x.transpose(1, 2)  # [B, T, hidden_dim]
        x = self.final_layer_norm(x)
        
        return x, state
    
    @classmethod
    def from_v1_backbone(cls, v1_backbone: nn.Module, rope_dim: int = 64) -> "CausalVocosBackbone":
        """
        Create CausalVocosBackbone from NeuCodec V1 VocosBackbone.
        
        Copies all weights while converting to causal architecture.
        """
        backbone = cls(
            hidden_dim=v1_backbone.embed.in_channels,
            depth=len(v1_backbone.transformers),
            heads=v1_backbone.transformers[0].n_heads,
            rope_dim=rope_dim,
        )
        
        # Copy embed conv weights
        backbone.embed.conv.weight.data.copy_(v1_backbone.embed.weight.data)
        backbone.embed.conv.bias.data.copy_(v1_backbone.embed.bias.data)
        
        # Copy prior net
        for i, v1_block in enumerate(v1_backbone.prior_net):
            backbone.prior_net[i] = CausalResnetBlock.from_v1_block(v1_block)
        
        # Copy transformers
        for i, v1_transformer in enumerate(v1_backbone.transformers):
            backbone.transformers[i] = CausalTransformerBlock.from_v1_block(v1_transformer, rope_dim)
        
        # Copy post net
        for i, v1_block in enumerate(v1_backbone.post_net):
            backbone.post_net[i] = CausalResnetBlock.from_v1_block(v1_block)
        
        # Copy final layer norm
        backbone.final_layer_norm.load_state_dict(v1_backbone.final_layer_norm.state_dict())
        
        return backbone


class StreamingCodecDecoder(nn.Module):
    """
    Streaming Codec Decoder for Audar-Codec.
    
    Complete streaming decoder that converts FSQ codes to audio.
    Supports both full-sequence (training) and streaming (inference) modes.
    
    Architecture:
    - quantizer: ResidualFSQ (from V1, frozen)
    - fc_post_a: Linear projection (2048 -> 1024)
    - backbone: CausalVocosBackbone
    - head: StreamingISTFTHead
    
    Args:
        hidden_dim: Hidden dimension (1024)
        depth: Number of transformer layers (12)
        heads: Number of attention heads (16)
        rope_dim: RoPE dimension (64)
        hop_length: Hop length for ISTFT (480 for 24kHz)
        vq_dim: VQ dimension (2048)
    """
    
    def __init__(
        self,
        hidden_dim: int = 1024,
        depth: int = 12,
        heads: int = 16,
        rope_dim: int = 64,
        hop_length: int = 480,
        vq_dim: int = 2048,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.hop_length = hop_length
        self.vq_dim = vq_dim
        
        # Quantizer will be loaded from V1 (frozen)
        self.quantizer = None
        
        # Post-quantization projection (2048 -> 1024)
        self.fc_post_a = nn.Linear(vq_dim, hidden_dim)
        
        # Causal backbone
        self.backbone = CausalVocosBackbone(
            hidden_dim=hidden_dim,
            depth=depth,
            heads=heads,
            rope_dim=rope_dim,
        )
        
        # Streaming ISTFT head
        self.head = StreamingISTFTHead(
            dim=hidden_dim,
            n_fft=hop_length * 4,
            hop_length=hop_length,
            padding="same",
        )
    
    def set_quantizer(self, quantizer: nn.Module):
        """Set the quantizer module (from V1)."""
        self.quantizer = quantizer
        self.quantizer.eval()
        for param in self.quantizer.parameters():
            param.requires_grad = False
    
    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """
        Convert FSQ codes to embeddings.
        
        Args:
            codes: FSQ codes [B, 1, T] or [B, T, 1]
            
        Returns:
            embeddings: [B, T, vq_dim]
        """
        if self.quantizer is None:
            raise RuntimeError("Quantizer not set. Call set_quantizer() first.")
        
        # Ensure correct shape [B, T, 1] for get_output_from_indices
        if codes.shape[1] == 1:
            codes = codes.transpose(1, 2)  # [B, 1, T] -> [B, T, 1]
        
        with torch.no_grad():
            emb = self.quantizer.get_output_from_indices(codes)
        
        return emb  # [B, T, vq_dim]
    
    def forward(
        self,
        x: torch.Tensor,
        from_codes: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Full sequence forward (training mode).
        
        Args:
            x: Input embeddings [B, T, vq_dim] or codes [B, 1, T] if from_codes=True
            from_codes: Whether input is FSQ codes
            
        Returns:
            audio: Reconstructed waveform [B, 1, samples]
            spec: Predicted spectrogram features
        """
        # Decode codes if needed
        if from_codes:
            x = self.decode_codes(x)  # [B, T, vq_dim]
        
        # Project to hidden dim
        x = self.fc_post_a(x)  # [B, T, hidden_dim]
        
        # Backbone
        x = self.backbone(x)  # [B, T, hidden_dim]
        
        # ISTFT head
        audio, spec = self.head(x)
        
        return audio, spec
    
    def forward_streaming(
        self,
        x: torch.Tensor,
        state: Optional[DecoderStreamingState] = None,
        from_codes: bool = False,
    ) -> Tuple[torch.Tensor, DecoderStreamingState]:
        """
        Streaming forward with state.
        
        Args:
            x: Input chunk [B, T_chunk, vq_dim] or codes [B, 1, T_chunk] if from_codes=True
            state: Streaming state from previous call
            from_codes: Whether input is FSQ codes
            
        Returns:
            audio: Output audio chunk [B, 1, T_chunk * hop_length]
            state: Updated streaming state
        """
        if state is None:
            state = DecoderStreamingState()
        
        # Decode codes if needed
        if from_codes:
            x = self.decode_codes(x)  # [B, T, vq_dim]
        
        # Project to hidden dim
        x = self.fc_post_a(x)  # [B, T, hidden_dim]
        
        # Backbone (streaming)
        x, state = self.backbone.forward_streaming(x, state)  # [B, T, hidden_dim]
        
        # ISTFT head (streaming)
        audio, state.istft_buffer = self.head.forward_streaming(x, state.istft_buffer)
        
        return audio, state
    
    @classmethod
    def from_v1_decoder(
        cls,
        v1_decoder: nn.Module,
        fc_post_a: nn.Module,
        rope_dim: int = 64,
    ) -> "StreamingCodecDecoder":
        """
        Create StreamingCodecDecoder from NeuCodec V1 components.
        
        Args:
            v1_decoder: CodecDecoderVocos from V1
            fc_post_a: fc_post_a Linear layer from NeuCodec
            rope_dim: RoPE dimension
            
        Returns:
            Streaming decoder with copied weights
        """
        decoder = cls(
            hidden_dim=v1_decoder.backbone.embed.in_channels,
            depth=len(v1_decoder.backbone.transformers),
            heads=v1_decoder.backbone.transformers[0].n_heads,
            rope_dim=rope_dim,
            hop_length=v1_decoder.hop_length,
            vq_dim=fc_post_a.in_features,
        )
        
        # Set quantizer (frozen)
        decoder.set_quantizer(v1_decoder.quantizer)
        
        # Copy fc_post_a
        decoder.fc_post_a.load_state_dict(fc_post_a.state_dict())
        
        # Convert backbone
        decoder.backbone = CausalVocosBackbone.from_v1_backbone(
            v1_decoder.backbone, rope_dim
        )
        
        # Copy ISTFT head weights
        decoder.head.out.load_state_dict(v1_decoder.head.out.state_dict())
        
        return decoder
    
    def get_initial_state(self) -> DecoderStreamingState:
        """Create initial streaming state."""
        return DecoderStreamingState()
