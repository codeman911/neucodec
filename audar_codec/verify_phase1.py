#!/usr/bin/env python3
"""
Audar-Codec Phase 1 Verification Script

Run with: poetry run python audar_codec/verify_phase1.py
"""

import sys
import torch

def main():
    print("=" * 60)
    print("Audar-Codec Phase 1 Verification")
    print("=" * 60)
    
    # Test imports
    print("\n[1/6] Testing imports...")
    try:
        from audar_codec.streaming.cache import KVCache, ConvCache
        from audar_codec.streaming.causal_layers import CausalConv1d, RMSNorm
        from audar_codec.streaming.istft_streaming import StreamingISTFT, StreamingISTFTHead
        from audar_codec.core.streaming_decoder import (
            CausalResnetBlock,
            CausalVocosBackbone,
            StreamingCodecDecoder,
        )
        from audar_codec.model import AudarCodec, AudarCodecConfig
        print("  All imports successful")
    except ImportError as e:
        print(f"  FAILED: {e}")
        return 1
    
    # Test KVCache
    print("\n[2/6] Testing KVCache...")
    cache = KVCache()
    k = torch.randn(2, 8, 5, 64)
    v = torch.randn(2, 8, 5, 64)
    k_out, v_out = cache.update(k, v)
    assert k_out.shape == k.shape, f"Expected {k.shape}, got {k_out.shape}"
    assert cache.get_seq_length() == 5
    k2, v2 = torch.randn(2, 8, 3, 64), torch.randn(2, 8, 3, 64)
    k_out, v_out = cache.update(k2, v2)
    assert cache.get_seq_length() == 8, f"Expected 8, got {cache.get_seq_length()}"
    print("  KVCache OK")
    
    # Test CausalConv1d
    print("\n[3/6] Testing CausalConv1d...")
    conv = CausalConv1d(64, 128, kernel_size=3)
    x = torch.randn(2, 64, 20)
    out = conv(x)
    assert out.shape == (2, 128, 20), f"Expected (2, 128, 20), got {out.shape}"
    # Test streaming
    conv.eval()
    cache = None
    streaming_outs = []
    with torch.no_grad():
        for i in range(4):
            chunk = x[:, :, i*5:(i+1)*5]
            out, cache = conv.forward_streaming(chunk, cache)
            streaming_outs.append(out)
    streaming_out = torch.cat(streaming_outs, dim=2)
    assert streaming_out.shape == (2, 128, 20)
    print("  CausalConv1d OK")
    
    # Test CausalVocosBackbone
    print("\n[4/6] Testing CausalVocosBackbone...")
    backbone = CausalVocosBackbone(hidden_dim=256, depth=2, heads=4, rope_dim=32)
    x = torch.randn(2, 20, 256)
    out = backbone(x)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"
    # Test streaming
    backbone.eval()
    state = None
    with torch.no_grad():
        out, state = backbone.forward_streaming(torch.randn(1, 5, 256), state)
    assert out.shape == (1, 5, 256)
    print("  CausalVocosBackbone OK")
    
    # Test StreamingCodecDecoder
    print("\n[5/6] Testing StreamingCodecDecoder...")
    decoder = StreamingCodecDecoder(
        hidden_dim=256,
        depth=2,
        heads=4,
        rope_dim=32,
        hop_length=256,
        vq_dim=512,
    )
    x = torch.randn(2, 20, 512)
    audio, spec = decoder(x, from_codes=False)
    assert audio.shape[0] == 2
    assert audio.shape[1] == 1
    expected_samples = 20 * 256
    assert audio.shape[2] == expected_samples, f"Expected {expected_samples}, got {audio.shape[2]}"
    print(f"  Full sequence output: {audio.shape}")
    
    # Test streaming
    decoder.eval()
    state = None
    chunk = torch.randn(1, 5, 512)
    with torch.no_grad():
        audio, state = decoder.forward_streaming(chunk, state, from_codes=False)
    assert audio.shape == (1, 1, 5 * 256), f"Expected (1, 1, 1280), got {audio.shape}"
    print(f"  Streaming output: {audio.shape}")
    print("  StreamingCodecDecoder OK")
    
    # Test AudarCodec
    print("\n[6/6] Testing AudarCodec...")
    config = AudarCodecConfig(
        hidden_dim=256,
        depth=2,
        heads=4,
        rope_dim=32,
        hop_length=256,
        vq_dim=512,
    )
    codec = AudarCodec(config)
    codec.eval()
    
    # Test streaming decode
    state = codec.get_streaming_state()
    emb = torch.randn(1, 10, 512)
    with torch.no_grad():
        audio, state = codec.decode_embeddings_streaming(emb, state)
    assert audio.shape == (1, 1, 10 * 256)
    print(f"  AudarCodec streaming: {audio.shape}")
    print("  AudarCodec OK")
    
    print("\n" + "=" * 60)
    print("All Phase 1 verification tests PASSED!")
    print("=" * 60)
    
    # Print summary
    print("\nPhase 1 Implementation Summary:")
    print("- Streaming cache management (KVCache, ConvCache)")
    print("- Causal convolutions with left-only padding")
    print("- Causal attention with KV-cache")
    print("- Streaming ISTFT with overlap-add")
    print("- Full StreamingCodecDecoder")
    print("- AudarCodec main interface")
    print("\nNext steps:")
    print("1. Load V1 checkpoint: AudarCodec.from_v1_checkpoint('path/to/v1.bin')")
    print("2. Train Phase 1: Use configs/phase1_decoder.yaml")
    print("3. Benchmark: codec.streaming_benchmark()")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
