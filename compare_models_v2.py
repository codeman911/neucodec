#!/usr/bin/env python3
"""Compare base vs finetuned NeuCodec on Arabic audio samples."""

import os
import warnings
warnings.filterwarnings('ignore')

import torch
import torchaudio
import soundfile as sf
import numpy as np
from pathlib import Path

from neucodec.model import NeuCodec


def load_models():
    """Load base and finetuned models."""
    print("Loading base model...")
    base_model = NeuCodec.from_pretrained("neuphonic/neucodec")
    base_model.eval()
    
    print("Loading finetuned model (step 32500, val_loss=0.414)...")
    ft_model = NeuCodec.from_pretrained("neuphonic/neucodec")
    ckpt = torch.load("checkpoints/finetune/checkpoint_best.pt", map_location='cpu', weights_only=False)
    ft_model.load_state_dict(ckpt['generator'], strict=False)
    ft_model.eval()
    
    print(f"Checkpoint: step={ckpt['global_step']}, val_loss={ckpt['best_val_loss']:.4f}")
    
    return base_model, ft_model


def process_audio(model, audio_path):
    """Encode and decode audio through model."""
    data, sr = sf.read(audio_path)
    waveform = torch.from_numpy(data).float()
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # [T] -> [1, T]
    
    if sr != 16000:
        waveform = torchaudio.functional.resample(waveform, sr, 16000)
    
    waveform = waveform.unsqueeze(0)  # [1, T] -> [1, 1, T]
    
    with torch.no_grad():
        codes = model.encode_code(waveform)
        recon = model.decode_code(codes)
    
    return waveform.squeeze(), recon.squeeze(), codes


def compute_metrics(original, reconstructed, sr=16000):
    """Compute audio quality metrics."""
    # Resample reconstructed from 24kHz to 16kHz for comparison
    if reconstructed.shape[-1] != original.shape[-1]:
        resampler = torchaudio.transforms.Resample(24000, 16000)
        reconstructed = resampler(reconstructed.unsqueeze(0)).squeeze()
    
    min_len = min(len(original), len(reconstructed))
    orig = original[:min_len].numpy()
    recon = reconstructed[:min_len].numpy()
    
    # MSE
    mse = np.mean((orig - recon) ** 2)
    
    # SNR
    signal_power = np.mean(orig ** 2)
    noise_power = np.mean((orig - recon) ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    
    # Correlation
    corr = np.corrcoef(orig, recon)[0, 1]
    
    return {
        'mse': mse,
        'snr_db': snr,
        'correlation': corr,
        'orig_std': np.std(orig),
        'recon_std': np.std(recon),
        'std_ratio': np.std(recon) / (np.std(orig) + 1e-10)
    }


def main():
    base_model, ft_model = load_models()
    
    sample_dir = Path("sample_dataset_6")
    output_dir = Path("comparison_outputs_v2")
    output_dir.mkdir(exist_ok=True)
    
    # Test files
    test_files = [
        "emirati_female_1_v6.wav",
        "emirati_male_1_v6.wav",
        "saudi_female_1_v6.wav",
        "saudi_male_1_v6.wav",
    ]
    
    results = []
    
    print("\n" + "="*90)
    print(f"{'File':<25} {'Model':<12} {'SNR(dB)':<10} {'Corr':<8} {'STD Ratio':<10} {'MSE':<12}")
    print("="*90)
    
    for fname in test_files:
        audio_path = sample_dir / fname
        if not audio_path.exists():
            print(f"Skipping {fname} - not found")
            continue
        
        # Process with base model
        orig, base_recon, base_codes = process_audio(base_model, audio_path)
        base_metrics = compute_metrics(orig, base_recon)
        
        # Process with finetuned model
        _, ft_recon, ft_codes = process_audio(ft_model, audio_path)
        ft_metrics = compute_metrics(orig, ft_recon)
        
        # Token difference
        token_diff = (base_codes != ft_codes).float().mean().item() * 100
        
        print(f"{fname:<25} {'Base':<12} {base_metrics['snr_db']:>8.2f}  {base_metrics['correlation']:>6.4f}  {base_metrics['std_ratio']:>8.4f}  {base_metrics['mse']:.6f}")
        print(f"{'':<25} {'Finetuned':<12} {ft_metrics['snr_db']:>8.2f}  {ft_metrics['correlation']:>6.4f}  {ft_metrics['std_ratio']:>8.4f}  {ft_metrics['mse']:.6f}")
        print(f"{'':<25} {'Token Δ':<12} {token_diff:>7.1f}%")
        print("-"*90)
        
        results.append({
            'file': fname,
            'base': base_metrics,
            'ft': ft_metrics,
            'token_diff': token_diff
        })
        
        # Save audio files
        stem = Path(fname).stem
        sf.write(output_dir / f"{stem}_original.wav", orig.numpy(), 16000)
        sf.write(output_dir / f"{stem}_base_recon.wav", base_recon.numpy(), 24000)
        sf.write(output_dir / f"{stem}_ft_recon.wav", ft_recon.numpy(), 24000)
    
    # Summary
    print("\n" + "="*90)
    print("SUMMARY")
    print("="*90)
    
    avg_base_snr = np.mean([r['base']['snr_db'] for r in results])
    avg_ft_snr = np.mean([r['ft']['snr_db'] for r in results])
    avg_base_corr = np.mean([r['base']['correlation'] for r in results])
    avg_ft_corr = np.mean([r['ft']['correlation'] for r in results])
    avg_base_std = np.mean([r['base']['std_ratio'] for r in results])
    avg_ft_std = np.mean([r['ft']['std_ratio'] for r in results])
    avg_token_diff = np.mean([r['token_diff'] for r in results])
    
    print(f"{'Metric':<20} {'Base Model':<15} {'Finetuned':<15} {'Δ':<10}")
    print("-"*60)
    print(f"{'Avg SNR (dB)':<20} {avg_base_snr:<15.2f} {avg_ft_snr:<15.2f} {avg_ft_snr - avg_base_snr:>+.2f}")
    print(f"{'Avg Correlation':<20} {avg_base_corr:<15.4f} {avg_ft_corr:<15.4f} {avg_ft_corr - avg_base_corr:>+.4f}")
    print(f"{'Avg STD Ratio':<20} {avg_base_std:<15.4f} {avg_ft_std:<15.4f} {avg_ft_std - avg_base_std:>+.4f}")
    print(f"{'Avg Token Diff':<20} {'-':<15} {avg_token_diff:<14.1f}%")
    
    print(f"\nOutput files saved to: {output_dir}/")
    print("Listen to *_original.wav, *_base_recon.wav, *_ft_recon.wav to compare quality")


if __name__ == "__main__":
    main()
