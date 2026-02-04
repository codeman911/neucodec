#!/usr/bin/env python3
"""
Diagnose the sample rate mismatch bug in NeuCodec training.

This script demonstrates that:
1. NeuCodec input is 16kHz
2. NeuCodec output is 24kHz  
3. The training code incorrectly computes mel loss assuming both are same rate
"""

import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def test_sample_rates():
    """Test NeuCodec input/output sample rates."""
    print("="*70)
    print("NEUCODEC SAMPLE RATE ANALYSIS")
    print("="*70)
    
    from neucodec.model import NeuCodec
    
    print("\nLoading NeuCodec base model...")
    model = NeuCodec.from_pretrained("neuphonic/neucodec")
    model.eval()
    
    print(f"\nModel configured sample rates:")
    print(f"  model.sample_rate = {model.sample_rate}")  # Should be 24000
    print(f"  model.hop_length = {model.hop_length}")    # Should be 480
    
    # Create test audio at 16kHz (what training uses)
    duration_sec = 1.0
    input_sr = 16000
    output_sr = 24000
    
    num_input_samples = int(duration_sec * input_sr)
    test_audio = torch.randn(1, 1, num_input_samples)  # [B, 1, T] at 16kHz
    
    print(f"\n--- Test with {duration_sec}s audio ---")
    print(f"Input audio shape: {test_audio.shape}")
    print(f"Input samples: {num_input_samples} @ {input_sr}Hz")
    
    with torch.no_grad():
        # Encode
        codes = model.encode_code(test_audio)
        print(f"\nEncoded FSQ codes shape: {codes.shape}")
        print(f"Tokens per second: {codes.shape[-1] / duration_sec:.1f} (expected: 50)")
        
        # Decode
        recon = model.decode_code(codes)
        print(f"\nDecoded audio shape: {recon.shape}")
        print(f"Output samples: {recon.shape[-1]}")
        
        # Calculate output duration
        output_duration = recon.shape[-1] / output_sr
        print(f"Output duration @ 24kHz: {output_duration:.3f}s")
        
        # Show the mismatch
        print("\n" + "="*70)
        print("THE BUG IN TRAINING CODE:")
        print("="*70)
        print(f"""
In train.py lines 395-400:

    # Match lengths  <-- THIS IS WRONG!
    min_len = min(audio.shape[-1], audio_recon.shape[-1])
    audio = audio[..., :min_len]           # 16kHz input: {num_input_samples} samples
    audio_recon = audio_recon[..., :min_len]  # 24kHz output: {recon.shape[-1]} samples
    
    # Mel loss  <-- COMPUTED WITH WRONG SAMPLE RATE!
    mel_loss = self.mel_loss(audio_recon, audio)  # Loss uses 16kHz config!

The training truncates to min({num_input_samples}, {recon.shape[-1]}) = {min(num_input_samples, recon.shape[-1])} samples.

This means:
  - Input (16kHz):  {num_input_samples} samples = {duration_sec}s
  - Output (24kHz): {recon.shape[-1]} samples truncated to {min(num_input_samples, recon.shape[-1])} = {min(num_input_samples, recon.shape[-1])/output_sr:.3f}s @ 24kHz
  
The model is being trained to compress {duration_sec}s of content into {min(num_input_samples, recon.shape[-1])/output_sr:.3f}s!
This creates the "fast forward" effect.
""")
        
        # Show what the mel loss sees
        print("="*70)
        print("MEL SPECTROGRAM COMPARISON:")
        print("="*70)
        
        # The mel loss is configured for 16kHz but receives 24kHz audio
        print(f"""
The MultiResolutionMelLoss is initialized with:
    sample_rate=16000  (from config)
    
But it receives:
    audio_recon: 24kHz audio (from decoder)
    audio: 16kHz audio (from dataloader)

When computing mel spectrograms at the SAME frequencies for both:
  - 16kHz audio: Nyquist = 8kHz, frequencies map correctly
  - 24kHz audio: Nyquist = 12kHz, but mel bins computed as if 8kHz!
  
This means the model learns to produce audio where:
  - Speech at 4kHz (24kHz audio) gets mapped to mel bin for 2.67kHz
  - The fundamental frequency shifts up by 1.5x = "chipmunk" effect
  - Combined with temporal compression = "fast forward donald duck"
""")
        
        # Verify the expected output length
        expected_output_samples = int(codes.shape[-1] * model.hop_length)
        print(f"\nExpected output length calculation:")
        print(f"  tokens ({codes.shape[-1]}) x hop_length ({model.hop_length}) = {expected_output_samples}")
        print(f"  Actual output: {recon.shape[-1]}")
        print(f"  At 24kHz, this is {expected_output_samples/24000:.3f}s")


def show_fix():
    """Show how to fix the training code."""
    print("\n" + "="*70)
    print("RECOMMENDED FIX:")
    print("="*70)
    print("""
Option 1: Resample output to match input (simpler)
-------------------------------------------------
In train.py, after decode_code():

    import torchaudio.transforms as T
    
    # Resample 24kHz output to 16kHz for loss computation
    resampler = T.Resample(24000, 16000).to(audio_recon.device)
    audio_recon_16k = resampler(audio_recon)
    
    # Now compute mel loss at same sample rate
    mel_loss = self.mel_loss(audio_recon_16k, audio)


Option 2: Compute loss at 24kHz (better quality)
------------------------------------------------
1. Resample input to 24kHz before loss:
    
    resampler = T.Resample(16000, 24000).to(audio.device)
    audio_24k = resampler(audio)
    
2. Update mel loss config to use 24kHz:
    
    self.mel_loss = MultiResolutionMelLoss(
        sample_rate=24000,  # Changed from 16000
        ...
    )
    
3. Compute loss:
    mel_loss = self.mel_loss(audio_recon, audio_24k)


Option 3: Use the same sample rate throughout (cleanest)
-------------------------------------------------------
Train with 24kHz input audio:
- Resample all training data to 24kHz
- Set data.sample_rate: 24000 in config
- This avoids any resampling during training
""")


if __name__ == "__main__":
    test_sample_rates()
    show_fix()
