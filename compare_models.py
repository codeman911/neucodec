#!/usr/bin/env python3
"""
Compare base NeuCodec model vs finetuned model on Arabic audio samples.
"""

import os
import torch
import torchaudio
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

from neucodec import NeuCodec


def load_audio(audio_path: str, target_sr: int = 16000):
    """Load and preprocess audio file."""
    waveform, sr = torchaudio.load(audio_path)
    
    # Resample if needed
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(sr, target_sr)
        waveform = resampler(waveform)
    
    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    return waveform


def process_audio(model, waveform, device="cpu"):
    """Encode and decode audio through model."""
    waveform = waveform.unsqueeze(0).to(device)  # Add batch dim
    
    with torch.no_grad():
        codes = model.encode_code(waveform)
        reconstructed = model.decode_code(codes)
    
    return codes, reconstructed.squeeze(0).cpu()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="comparison_outputs")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()
    
    device = args.device
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load base model
    print("Loading base NeuCodec model...")
    base_model = NeuCodec.from_pretrained("neuphonic/neucodec")
    base_model = base_model.to(device)
    base_model.eval()
    
    # Load finetuned model
    print(f"Loading finetuned model from {args.checkpoint}...")
    finetuned_model = NeuCodec.from_pretrained("neuphonic/neucodec")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Load weights (handle both naming conventions)
    state_dict = checkpoint.get("generator", checkpoint.get("generator_state_dict"))
    if state_dict:
        # Handle DDP prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k[7:] if k.startswith("module.") else k
            new_state_dict[new_key] = v
        finetuned_model.load_state_dict(new_state_dict, strict=False)
    
    finetuned_model = finetuned_model.to(device)
    finetuned_model.eval()
    
    # Find audio files
    input_dir = Path(args.input_dir)
    audio_files = list(input_dir.glob("*.wav"))[:4]  # Limit to 4 files
    
    print(f"\nProcessing {len(audio_files)} audio files...")
    print("="*70)
    
    results = []
    
    for audio_path in audio_files:
        print(f"\n>>> {audio_path.name}")
        
        # Load audio
        waveform = load_audio(str(audio_path))
        duration = waveform.shape[-1] / 16000
        print(f"    Duration: {duration:.2f}s, Shape: {waveform.shape}")
        
        # Save original (resampled to 16kHz)
        orig_path = output_dir / f"{audio_path.stem}_original.wav"
        torchaudio.save(str(orig_path), waveform, 16000)
        
        # Process with base model
        base_codes, base_recon = process_audio(base_model, waveform, device)
        base_path = output_dir / f"{audio_path.stem}_base.wav"
        torchaudio.save(str(base_path), base_recon, 24000)
        print(f"    Base model:     tokens={base_codes.shape}, recon={base_recon.shape}")
        
        # Process with finetuned model
        ft_codes, ft_recon = process_audio(finetuned_model, waveform, device)
        ft_path = output_dir / f"{audio_path.stem}_finetuned.wav"
        torchaudio.save(str(ft_path), ft_recon, 24000)
        print(f"    Finetuned:      tokens={ft_codes.shape}, recon={ft_recon.shape}")
        
        # Compare tokens
        token_diff = (base_codes != ft_codes).float().mean().item() * 100
        print(f"    Token diff:     {token_diff:.1f}%")
        
        # Compare reconstructions (MSE)
        min_len = min(base_recon.shape[-1], ft_recon.shape[-1])
        mse = torch.nn.functional.mse_loss(
            base_recon[..., :min_len], 
            ft_recon[..., :min_len]
        ).item()
        print(f"    Recon MSE:      {mse:.6f}")
        
        results.append({
            "file": audio_path.name,
            "duration": duration,
            "token_diff_pct": token_diff,
            "recon_mse": mse,
        })
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'File':<30} {'Duration':>10} {'Token Diff':>12} {'MSE':>12}")
    print("-"*70)
    for r in results:
        print(f"{r['file']:<30} {r['duration']:>10.2f}s {r['token_diff_pct']:>11.1f}% {r['recon_mse']:>12.6f}")
    
    avg_diff = sum(r['token_diff_pct'] for r in results) / len(results)
    avg_mse = sum(r['recon_mse'] for r in results) / len(results)
    print("-"*70)
    print(f"{'AVERAGE':<30} {'':<10} {avg_diff:>11.1f}% {avg_mse:>12.6f}")
    
    print(f"\nOutputs saved to: {output_dir}/")
    print("Files generated per sample:")
    print("  - *_original.wav   (16kHz input)")
    print("  - *_base.wav       (24kHz base model reconstruction)")
    print("  - *_finetuned.wav  (24kHz finetuned model reconstruction)")


if __name__ == "__main__":
    main()
