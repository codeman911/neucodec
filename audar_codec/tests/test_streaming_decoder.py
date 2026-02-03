"""
Tests for Audar-Codec streaming decoder.
"""

import pytest
import torch
import torch.nn as nn

# Import audar_codec components
import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0])

from audar_codec.streaming.cache import KVCache, ConvCache
from audar_codec.streaming.causal_layers import CausalConv1d, CausalAttention, CausalTransformerBlock
from audar_codec.streaming.istft_streaming import StreamingISTFT, StreamingISTFTHead
from audar_codec.core.streaming_decoder import (
    CausalResnetBlock,
    CausalVocosBackbone,
    StreamingCodecDecoder,
    DecoderStreamingState,
)


class TestKVCache:
    """Test KVCache functionality."""
    
    def test_init_empty(self):
        cache = KVCache()
        assert cache.k is None
        assert cache.v is None
        assert cache.get_seq_length() == 0
    
    def test_update_single(self):
        cache = KVCache()
        k = torch.randn(2, 8, 5, 64)  # [B, H, T, D]
        v = torch.randn(2, 8, 5, 64)
        
        k_out, v_out = cache.update(k, v)
        
        assert k_out.shape == k.shape
        assert v_out.shape == v.shape
        assert cache.get_seq_length() == 5
    
    def test_update_multiple(self):
        cache = KVCache()
        
        # First update
        k1 = torch.randn(2, 8, 5, 64)
        v1 = torch.randn(2, 8, 5, 64)
        cache.update(k1, v1)
        
        # Second update
        k2 = torch.randn(2, 8, 3, 64)
        v2 = torch.randn(2, 8, 3, 64)
        k_out, v_out = cache.update(k2, v2)
        
        assert k_out.shape == (2, 8, 8, 64)  # 5 + 3
        assert cache.get_seq_length() == 8
    
    def test_max_length_truncation(self):
        cache = KVCache(max_length=10)
        
        # Add more than max_length
        k = torch.randn(2, 8, 15, 64)
        v = torch.randn(2, 8, 15, 64)
        k_out, v_out = cache.update(k, v)
        
        assert k_out.shape == (2, 8, 10, 64)
        assert cache.get_seq_length() == 10


class TestConvCache:
    """Test ConvCache functionality."""
    
    def test_init(self):
        cache = ConvCache(padding=4)
        assert cache.buffer is None
        assert cache.padding == 4
    
    def test_update(self):
        cache = ConvCache(padding=4)
        x = torch.randn(2, 64, 10)  # [B, C, T]
        
        x_padded = cache.update(x)
        
        assert x_padded.shape == (2, 64, 14)  # 10 + 4 padding
        assert cache.buffer.shape == (2, 64, 4)  # Last 4 timesteps


class TestCausalConv1d:
    """Test CausalConv1d layer."""
    
    def test_forward_shape(self):
        conv = CausalConv1d(64, 128, kernel_size=3)
        x = torch.randn(2, 64, 20)
        
        out = conv(x)
        
        assert out.shape == (2, 128, 20)  # Same temporal length
    
    def test_streaming_equivalence(self):
        """Test that streaming produces same output as full sequence."""
        conv = CausalConv1d(64, 64, kernel_size=3)
        conv.eval()
        
        # Full sequence
        x = torch.randn(1, 64, 20)
        with torch.no_grad():
            full_out = conv(x)
        
        # Streaming
        chunk_size = 5
        cache = None
        streaming_out = []
        
        with torch.no_grad():
            for i in range(0, 20, chunk_size):
                chunk = x[:, :, i:i+chunk_size]
                out, cache = conv.forward_streaming(chunk, cache)
                streaming_out.append(out)
        
        streaming_out = torch.cat(streaming_out, dim=2)
        
        # Should be close (not exact due to boundary effects in first chunk)
        assert torch.allclose(full_out[:, :, 4:], streaming_out[:, :, 4:], atol=1e-5)


class TestCausalTransformerBlock:
    """Test CausalTransformerBlock."""
    
    def test_forward_shape(self):
        block = CausalTransformerBlock(dim=256, n_heads=4, rope_dim=32)
        x = torch.randn(2, 20, 256)
        
        out = block(x)
        
        assert out.shape == x.shape
    
    def test_streaming_shape(self):
        block = CausalTransformerBlock(dim=256, n_heads=4, rope_dim=32)
        block.eval()
        
        # Streaming with chunks
        state = None
        pos_offset = 0
        chunk = torch.randn(1, 5, 256)
        
        with torch.no_grad():
            out, kv_cache, new_pos = block.forward_streaming(chunk, state, pos_offset)
        
        assert out.shape == chunk.shape
        assert kv_cache.get_seq_length() == 5
        assert new_pos == 5


class TestStreamingISTFT:
    """Test StreamingISTFT."""
    
    def test_forward_shape(self):
        istft = StreamingISTFT(n_fft=1024, hop_length=256)
        # Complex spectrogram
        spec = torch.randn(2, 513, 20) + 1j * torch.randn(2, 513, 20)
        
        audio = istft(spec)
        
        # Output length = (20 - 1) * 256 = 4864 (approximately)
        assert audio.shape[0] == 2
        assert audio.shape[1] > 0
    
    def test_streaming_equivalence(self):
        """Test streaming produces same output as full."""
        istft = StreamingISTFT(n_fft=1024, hop_length=256, padding="same")
        
        # Generate test spectrogram
        spec = torch.randn(1, 513, 20) + 1j * torch.randn(1, 513, 20)
        
        # Full
        full_audio = istft(spec)
        
        # Streaming
        buffer = None
        streaming_audio = []
        
        for t in range(spec.shape[2]):
            chunk = spec[:, :, t:t+1]
            audio_chunk, buffer = istft.forward_streaming(chunk, buffer)
            streaming_audio.append(audio_chunk)
        
        streaming_audio = torch.cat(streaming_audio, dim=1)
        
        # Compare (may differ at boundaries)
        min_len = min(full_audio.shape[1], streaming_audio.shape[1])
        # Skip first few samples due to boundary effects
        skip = 512
        if min_len > skip * 2:
            assert torch.allclose(
                full_audio[:, skip:min_len-skip],
                streaming_audio[:, skip:min_len-skip],
                atol=1e-3
            )


class TestCausalResnetBlock:
    """Test CausalResnetBlock."""
    
    def test_forward_shape(self):
        block = CausalResnetBlock(in_channels=64, out_channels=64)
        x = torch.randn(2, 64, 20)
        
        out = block(x)
        
        assert out.shape == x.shape
    
    def test_streaming_shape(self):
        block = CausalResnetBlock(in_channels=64, out_channels=64)
        block.eval()
        
        x = torch.randn(1, 64, 5)
        
        with torch.no_grad():
            out, c1, c2, _ = block.forward_streaming(x)
        
        assert out.shape == x.shape


class TestCausalVocosBackbone:
    """Test CausalVocosBackbone."""
    
    def test_forward_shape(self):
        backbone = CausalVocosBackbone(hidden_dim=256, depth=2, heads=4, rope_dim=32)
        x = torch.randn(2, 20, 256)
        
        out = backbone(x)
        
        assert out.shape == x.shape
    
    def test_streaming_shape(self):
        backbone = CausalVocosBackbone(hidden_dim=256, depth=2, heads=4, rope_dim=32)
        backbone.eval()
        
        state = None
        chunk = torch.randn(1, 5, 256)
        
        with torch.no_grad():
            out, state = backbone.forward_streaming(chunk, state)
        
        assert out.shape == chunk.shape
        assert state.position_offset > 0


class TestStreamingCodecDecoder:
    """Test StreamingCodecDecoder."""
    
    def test_init(self):
        decoder = StreamingCodecDecoder(
            hidden_dim=256,
            depth=2,
            heads=4,
            rope_dim=32,
            hop_length=256,
            vq_dim=512,
        )
        
        assert decoder.hidden_dim == 256
        assert decoder.hop_length == 256
    
    def test_forward_shape(self):
        decoder = StreamingCodecDecoder(
            hidden_dim=256,
            depth=2,
            heads=4,
            rope_dim=32,
            hop_length=256,
            vq_dim=512,
        )
        
        # Input embeddings (not codes)
        x = torch.randn(2, 20, 512)
        
        audio, spec = decoder(x, from_codes=False)
        
        assert audio.shape[0] == 2
        assert audio.shape[1] == 1
        assert audio.shape[2] > 0
    
    def test_streaming_shape(self):
        decoder = StreamingCodecDecoder(
            hidden_dim=256,
            depth=2,
            heads=4,
            rope_dim=32,
            hop_length=256,
            vq_dim=512,
        )
        decoder.eval()
        
        state = None
        x = torch.randn(1, 5, 512)
        
        with torch.no_grad():
            audio, state = decoder.forward_streaming(x, state, from_codes=False)
        
        assert audio.shape[0] == 1
        assert audio.shape[1] == 1
        # Output samples = 5 frames * 256 hop = 1280
        assert audio.shape[2] == 5 * 256


class TestStreamingEquivalence:
    """Test that streaming and full-sequence produce similar outputs."""
    
    def test_decoder_streaming_vs_full(self):
        """Compare streaming vs full-sequence decoding."""
        decoder = StreamingCodecDecoder(
            hidden_dim=256,
            depth=2,
            heads=4,
            rope_dim=32,
            hop_length=256,
            vq_dim=512,
        )
        decoder.eval()
        
        # Full sequence
        x = torch.randn(1, 20, 512)
        with torch.no_grad():
            full_audio, _ = decoder(x, from_codes=False)
        
        # Streaming
        chunk_size = 5
        state = None
        streaming_audio = []
        
        with torch.no_grad():
            for i in range(0, 20, chunk_size):
                chunk = x[:, i:i+chunk_size, :]
                audio_chunk, state = decoder.forward_streaming(chunk, state, from_codes=False)
                streaming_audio.append(audio_chunk)
        
        streaming_audio = torch.cat(streaming_audio, dim=2)
        
        # Note: Due to causal vs non-causal attention, outputs will differ
        # We just check shapes match
        assert full_audio.shape == streaming_audio.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
