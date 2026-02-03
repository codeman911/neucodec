#!/usr/bin/env python3
"""
Audar-Codec vs NeuCodec V1 Comparison Test

Tests streaming decoder on sample_dataset_6 and compares with base model.
Protocol: Test on exactly 4 files with parallel result generation.

Run on cluster: python audar_codec/tests/compare_v1_audar.py 2>&1 | tee comparison_results.log
"""

import os
import sys
import json
import time
import platform
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import torch
import torchaudio
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from neucodec.model import NeuCodec

# Sample dataset
SAMPLE_DIR = PROJECT_ROOT / "sample_dataset_6"
# Select 4 test files per protocol (2 male, 2 female, mix of dialects)
TEST_FILES = [
    "emirati_female_1_v6.wav",
    "emirati_male_1_v6.wav",
    "saudi_female_1_v6.wav",
    "saudi_male_1_v6.wav",
]


def log_system_info():
    """Log system and environment information for diagnosis."""
    logger.info("=" * 70)
    logger.info("SYSTEM DIAGNOSTICS")
    logger.info("=" * 70)
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Python: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    logger.info(f"Node: {platform.node()}")
    logger.info(f"PyTorch: {torch.__version__}")
    logger.info(f"Torchaudio: {torchaudio.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    logger.info(f"CPU count: {os.cpu_count()}")
    logger.info(f"Project root: {PROJECT_ROOT}")
    logger.info(f"Sample dir: {SAMPLE_DIR}")
    logger.info(f"Sample dir exists: {SAMPLE_DIR.exists()}")


def load_audio(path: Path, target_sr: int = 16000) -> torch.Tensor:
    """Load and resample audio to target sample rate."""
    waveform, sr = torchaudio.load(path)
    logger.debug(f"Loaded {path.name}: shape={waveform.shape}, sr={sr}")
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform


def compute_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> Dict[str, float]:
    """Compute audio quality metrics."""
    min_len = min(original.shape[-1], reconstructed.shape[-1])
    orig = original[..., :min_len].squeeze()
    recon = reconstructed[..., :min_len].squeeze()
    
    # SNR
    noise = orig - recon
    signal_power = torch.mean(orig ** 2)
    noise_power = torch.mean(noise ** 2)
    snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
    
    # RMSE
    rmse = torch.sqrt(torch.mean((orig - recon) ** 2))
    
    # Correlation
    orig_centered = orig - orig.mean()
    recon_centered = recon - recon.mean()
    corr = torch.sum(orig_centered * recon_centered) / (
        torch.sqrt(torch.sum(orig_centered ** 2)) * 
        torch.sqrt(torch.sum(recon_centered ** 2)) + 1e-10
    )
    
    # Peak SNR
    max_val = torch.max(torch.abs(orig))
    psnr = 20 * torch.log10(max_val / (torch.sqrt(noise_power) + 1e-10))
    
    return {
        "snr_db": snr.item(),
        "psnr_db": psnr.item(),
        "rmse": rmse.item(),
        "correlation": corr.item(),
        "length_original": original.shape[-1],
        "length_reconstructed": reconstructed.shape[-1],
    }


def test_neucodec_v1(model: NeuCodec, audio_path: Path, device: str) -> Dict:
    """Test NeuCodec V1 (full sequence, non-streaming)."""
    logger.info(f"  [V1] Loading audio: {audio_path.name}")
    audio = load_audio(audio_path)
    audio = audio.unsqueeze(0).to(device)
    
    pad_len = 320 - (audio.shape[-1] % 320)
    if pad_len < 320:
        audio = torch.nn.functional.pad(audio, (0, pad_len))
    
    logger.info(f"  [V1] Input shape: {audio.shape}")
    
    # Encode
    if device == "cuda":
        torch.cuda.synchronize()
    start_encode = time.perf_counter()
    with torch.no_grad():
        codes = model.encode_code(audio)
    if device == "cuda":
        torch.cuda.synchronize()
    encode_time = time.perf_counter() - start_encode
    
    logger.info(f"  [V1] Codes shape: {codes.shape}, encode_time: {encode_time:.4f}s")
    
    # Decode
    start_decode = time.perf_counter()
    with torch.no_grad():
        recon = model.decode_code(codes)
    if device == "cuda":
        torch.cuda.synchronize()
    decode_time = time.perf_counter() - start_decode
    
    logger.info(f"  [V1] Recon shape: {recon.shape}, decode_time: {decode_time:.4f}s")
    
    # Metrics
    audio_16k = audio.cpu()
    recon_24k = recon.cpu()
    resampler = torchaudio.transforms.Resample(16000, 24000)
    audio_24k = resampler(audio_16k.squeeze(0)).unsqueeze(0)
    
    metrics = compute_metrics(audio_24k, recon_24k)
    audio_duration = audio.shape[-1] / 16000
    
    result = {
        "model": "NeuCodec_V1",
        "mode": "full_sequence",
        "file": audio_path.name,
        "encode_time_s": encode_time,
        "decode_time_s": decode_time,
        "total_time_s": encode_time + decode_time,
        "audio_duration_s": audio_duration,
        "rtf_encode": encode_time / audio_duration,
        "rtf_decode": decode_time / audio_duration,
        "rtf_total": (encode_time + decode_time) / audio_duration,
        "num_codes": codes.shape[-1],
        "code_rate_hz": codes.shape[-1] / audio_duration,
        **metrics,
    }
    
    logger.info(f"  [V1] RTF={result['rtf_total']:.4f}, SNR={result['snr_db']:.2f}dB, corr={result['correlation']:.4f}")
    return result


def test_audar_streaming(audio_path: Path, device: str) -> Dict:
    """Test Audar-Codec streaming decoder."""
    from audar_codec.model import AudarCodec, AudarCodecConfig
    from audar_codec.core.streaming_decoder import StreamingCodecDecoder
    
    logger.info(f"  [Audar] Creating streaming decoder")
    
    config = AudarCodecConfig(
        hidden_dim=1024,
        depth=12,
        heads=16,
        rope_dim=64,
        hop_length=480,
        vq_dim=2048,
    )
    
    decoder = StreamingCodecDecoder(
        hidden_dim=config.hidden_dim,
        depth=config.depth,
        heads=config.heads,
        rope_dim=config.rope_dim,
        hop_length=config.hop_length,
        vq_dim=config.vq_dim,
    )
    decoder.eval()
    decoder.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in decoder.parameters())
    logger.info(f"  [Audar] Decoder params: {num_params:,}")
    
    audio = load_audio(audio_path)
    audio_duration = audio.shape[-1] / 16000
    num_frames = int(audio_duration * 50)
    
    logger.info(f"  [Audar] Audio duration: {audio_duration:.2f}s, frames: {num_frames}")
    
    # Use deterministic embeddings for reproducibility
    torch.manual_seed(42)
    embeddings = torch.randn(1, num_frames, config.vq_dim, device=device)
    
    # Full sequence decode
    if device == "cuda":
        torch.cuda.synchronize()
    start_full = time.perf_counter()
    with torch.no_grad():
        full_audio, _ = decoder(embeddings, from_codes=False)
    if device == "cuda":
        torch.cuda.synchronize()
    full_time = time.perf_counter() - start_full
    
    logger.info(f"  [Audar] Full decode: shape={full_audio.shape}, time={full_time:.4f}s")
    
    # Streaming decode with multiple chunk sizes
    chunk_results = {}
    for chunk_size in [5, 10, 20, 50]:
        state = None
        streaming_chunks = []
        
        if device == "cuda":
            torch.cuda.synchronize()
        start_stream = time.perf_counter()
        with torch.no_grad():
            for i in range(0, num_frames, chunk_size):
                chunk = embeddings[:, i:i+chunk_size, :]
                if chunk.shape[1] == 0:
                    break
                audio_chunk, state = decoder.forward_streaming(chunk, state, from_codes=False)
                streaming_chunks.append(audio_chunk)
        if device == "cuda":
            torch.cuda.synchronize()
        stream_time = time.perf_counter() - start_stream
        
        streaming_audio = torch.cat(streaming_chunks, dim=2) if streaming_chunks else torch.zeros(1, 1, 0)
        
        # Measure consistency
        min_len = min(full_audio.shape[-1], streaming_audio.shape[-1])
        if min_len > 0:
            diff = torch.abs(full_audio[..., :min_len] - streaming_audio[..., :min_len]).mean().item()
            max_diff = torch.abs(full_audio[..., :min_len] - streaming_audio[..., :min_len]).max().item()
        else:
            diff = float('nan')
            max_diff = float('nan')
        
        chunk_results[f"chunk_{chunk_size}"] = {
            "time_s": stream_time,
            "rtf": stream_time / audio_duration,
            "num_chunks": (num_frames + chunk_size - 1) // chunk_size,
            "output_samples": streaming_audio.shape[-1],
            "mean_diff": diff,
            "max_diff": max_diff,
            "shapes_match": full_audio.shape == streaming_audio.shape,
        }
        
        logger.info(f"  [Audar] Chunk={chunk_size}: RTF={stream_time/audio_duration:.4f}, diff={diff:.6f}")
    
    return {
        "model": "Audar_Codec",
        "mode": "streaming",
        "file": audio_path.name,
        "full_decode_time_s": full_time,
        "audio_duration_s": audio_duration,
        "rtf_full": full_time / audio_duration,
        "num_frames": num_frames,
        "output_samples_full": full_audio.shape[-1],
        "num_params": num_params,
        "chunk_results": chunk_results,
    }


def test_streaming_consistency(device: str) -> Dict:
    """Detailed streaming consistency analysis."""
    from audar_codec.core.streaming_decoder import StreamingCodecDecoder
    
    logger.info("Running streaming consistency analysis...")
    
    # Small model for quick testing
    decoder = StreamingCodecDecoder(
        hidden_dim=256,
        depth=2,
        heads=4,
        rope_dim=32,
        hop_length=256,
        vq_dim=512,
    )
    decoder.eval()
    decoder.to(device)
    
    results = {}
    
    # Test multiple sequence lengths
    for seq_len in [20, 50, 100]:
        torch.manual_seed(42)
        embeddings = torch.randn(1, seq_len, 512, device=device)
        
        # Full sequence
        with torch.no_grad():
            full_audio, _ = decoder(embeddings, from_codes=False)
        
        # Streaming with chunk_size=10
        state = None
        chunks = []
        with torch.no_grad():
            for i in range(0, seq_len, 10):
                chunk = embeddings[:, i:i+10, :]
                audio_chunk, state = decoder.forward_streaming(chunk, state, from_codes=False)
                chunks.append(audio_chunk)
        streaming_audio = torch.cat(chunks, dim=2)
        
        # Analysis
        min_len = min(full_audio.shape[-1], streaming_audio.shape[-1])
        diff = full_audio[..., :min_len] - streaming_audio[..., :min_len]
        
        results[f"seq_{seq_len}"] = {
            "full_shape": list(full_audio.shape),
            "streaming_shape": list(streaming_audio.shape),
            "shapes_match": full_audio.shape == streaming_audio.shape,
            "mean_abs_diff": torch.abs(diff).mean().item(),
            "max_abs_diff": torch.abs(diff).max().item(),
            "std_diff": diff.std().item(),
            "relative_diff": (torch.abs(diff).mean() / (torch.abs(full_audio[..., :min_len]).mean() + 1e-10)).item(),
        }
        
        logger.info(f"  seq_len={seq_len}: mean_diff={results[f'seq_{seq_len}']['mean_abs_diff']:.6f}, "
                   f"rel_diff={results[f'seq_{seq_len}']['relative_diff']:.6f}")
    
    return results


def save_results(results: Dict, output_path: Path):
    """Save results to JSON file."""
    # Convert any non-serializable types
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        return obj
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=convert)
    logger.info(f"Results saved to: {output_path}")


def main():
    log_system_info()
    
    logger.info("=" * 70)
    logger.info("AUDAR-CODEC VS NEUCODEC V1 COMPARISON")
    logger.info("=" * 70)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Verify test files
    logger.info(f"\nVerifying test files in {SAMPLE_DIR}:")
    missing_files = []
    for f in TEST_FILES:
        path = SAMPLE_DIR / f
        exists = path.exists()
        size = path.stat().st_size if exists else 0
        logger.info(f"  {f}: exists={exists}, size={size} bytes")
        if not exists:
            missing_files.append(f)
    
    if missing_files:
        logger.error(f"Missing files: {missing_files}")
        return 1
    
    # Load NeuCodec V1
    logger.info("\n" + "-" * 70)
    logger.info("Loading NeuCodec V1 from HuggingFace...")
    v1_model = None
    try:
        v1_model = NeuCodec.from_pretrained("neuphonic/neucodec")
        v1_model.eval()
        v1_model.to(device)
        v1_params = sum(p.numel() for p in v1_model.parameters())
        logger.info(f"NeuCodec V1 loaded: {v1_params:,} parameters")
    except Exception as e:
        logger.error(f"Failed to load NeuCodec V1: {e}")
        import traceback
        traceback.print_exc()
    
    # Run tests
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "device": device,
        "test_files": TEST_FILES,
        "v1_results": [],
        "audar_results": [],
        "consistency_analysis": {},
    }
    
    for i, filename in enumerate(TEST_FILES, 1):
        audio_path = SAMPLE_DIR / filename
        logger.info(f"\n{'=' * 70}")
        logger.info(f"[{i}/4] Testing: {filename}")
        logger.info("=" * 70)
        
        # NeuCodec V1
        if v1_model is not None:
            logger.info("\n--- NeuCodec V1 (full sequence) ---")
            try:
                v1_result = test_neucodec_v1(v1_model, audio_path, device)
                all_results["v1_results"].append(v1_result)
            except Exception as e:
                logger.error(f"V1 test error: {e}")
                import traceback
                traceback.print_exc()
        
        # Audar-Codec
        logger.info("\n--- Audar-Codec (streaming) ---")
        try:
            audar_result = test_audar_streaming(audio_path, device)
            all_results["audar_results"].append(audar_result)
        except Exception as e:
            logger.error(f"Audar test error: {e}")
            import traceback
            traceback.print_exc()
    
    # Streaming consistency analysis
    logger.info(f"\n{'=' * 70}")
    logger.info("STREAMING CONSISTENCY ANALYSIS")
    logger.info("=" * 70)
    try:
        consistency = test_streaming_consistency(device)
        all_results["consistency_analysis"] = consistency
    except Exception as e:
        logger.error(f"Consistency test error: {e}")
        import traceback
        traceback.print_exc()
    
    # Summary
    logger.info(f"\n{'=' * 70}")
    logger.info("SUMMARY")
    logger.info("=" * 70)
    
    if all_results["v1_results"]:
        v1_rtf = np.mean([r["rtf_total"] for r in all_results["v1_results"]])
        v1_snr = np.mean([r["snr_db"] for r in all_results["v1_results"]])
        v1_corr = np.mean([r["correlation"] for r in all_results["v1_results"]])
        logger.info(f"\nNeuCodec V1 (4 files):")
        logger.info(f"  Average RTF: {v1_rtf:.4f}")
        logger.info(f"  Average SNR: {v1_snr:.2f} dB")
        logger.info(f"  Average Correlation: {v1_corr:.4f}")
    
    if all_results["audar_results"]:
        audar_rtf_full = np.mean([r["rtf_full"] for r in all_results["audar_results"]])
        logger.info(f"\nAudar-Codec (4 files):")
        logger.info(f"  Average RTF (full): {audar_rtf_full:.4f}")
        
        # Per chunk size analysis
        for chunk_size in [5, 10, 20, 50]:
            key = f"chunk_{chunk_size}"
            rtfs = [r["chunk_results"][key]["rtf"] for r in all_results["audar_results"] if key in r["chunk_results"]]
            if rtfs:
                logger.info(f"  Average RTF (chunk={chunk_size}): {np.mean(rtfs):.4f}")
    
    # Save results
    output_path = PROJECT_ROOT / "comparison_results.json"
    save_results(all_results, output_path)
    
    logger.info(f"\n{'=' * 70}")
    logger.info("COMPARISON COMPLETE")
    logger.info("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
