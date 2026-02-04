# Sample Rate Mismatch Analysis

## Overview

NeuCodec is designed to take **16kHz input** and produce **24kHz output** (upsampling codec). However, the current fine-tuning pipeline introduces critical sample rate mismatches that degrade audio quality.

## Architecture Reference

From NeuCodec Core Architecture:
- Input: 16kHz audio
- Output: 24kHz reconstructed audio
- FSQ operates at 50 tokens/sec (0.8 kbps)

## Issue 1: Loss Computation on Resampled Audio

### Location
- `training/train.py` lines 113-114, 398-405

### Code Analysis

```python
# Line 114: Resampler initialized in __init__
self.output_resampler = torchaudio.transforms.Resample(24000, 16000)

# Lines 395-405: In train_step()
# Decode (output is 24kHz)
audio_recon = gen_module.decode_code(fsq_codes)

# Resample 24kHz output to 16kHz to match input for loss computation
audio_recon = self.output_resampler.to(audio_recon.device)(audio_recon)

# Match lengths (now both are 16kHz)
min_len = min(audio.shape[-1], audio_recon.shape[-1])
audio = audio[..., :min_len]
audio_recon = audio_recon[..., :min_len]

# Mel loss computed on 16kHz
mel_loss = self.mel_loss(audio_recon, audio)
```

### Problem Description

1. **Downsampling destroys high-frequency information**: When 24kHz audio is resampled to 16kHz, all frequency content above 8kHz (Nyquist limit) is removed.

2. **No gradient for high-frequency reconstruction**: The model receives zero gradient signal for frequencies between 8-12kHz, which are present in 24kHz output but absent in the loss computation.

3. **Aliasing artifacts**: The resampling process can introduce aliasing artifacts that pollute the loss signal.

4. **Training-inference mismatch**: During training, the model is optimized for 16kHz quality, but during inference it produces 24kHz audio that was never properly supervised.

### Impact on Audio Quality

| Frequency Range | Training Supervision | Result |
|-----------------|---------------------|--------|
| 0-8kHz | Full supervision | Good quality |
| 8-12kHz | No supervision | Degraded/artifacts |

### Visualization

```
Training Pipeline (Current - PROBLEMATIC):
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Input 16kHz │───▶│   Encoder   │───▶│   Decoder   │───▶│ Output 24kHz│
└─────────────┘    └─────────────┘    └─────────────┘    └──────┬──────┘
                                                                │
                                                         Resample to 16kHz
                                                                │
                                                                ▼
┌─────────────┐                                         ┌─────────────┐
│ Target 16kHz│◀────────────────Loss Computation───────▶│ Recon 16kHz │
└─────────────┘                                         └─────────────┘
                        ⚠️ High frequencies LOST!
```

## Issue 2: Discriminator Sample Rate Mismatch

### Location
- `training/train.py` lines 414-439
- `training/configs/finetune_chatml.yaml` lines 148-166

### Configuration Analysis

```yaml
# finetune_chatml.yaml - Discriminator settings
discriminators:
  mpd:
    enabled: true
    periods: [2, 3, 5, 7, 11]  # Period-based patterns
    
  msd:
    enabled: true
    scales: 3
    
  ms_stft:
    enabled: true
    filters: 32
    n_ffts: [1024, 2048, 512]
    hop_lengths: [256, 512, 128]
    win_lengths: [1024, 2048, 512]
```

### Problem Description

1. **STFT parameters tuned for wrong sample rate**: The hop lengths and FFT sizes in MS-STFT discriminator determine the time-frequency resolution. At 16kHz vs 24kHz, these represent different real-world durations.

2. **MPD period mismatch**: Multi-Period Discriminator periods are designed for specific sample rates. At 24kHz, period=2 captures 12kHz patterns; at 16kHz, it captures 8kHz patterns.

3. **Adversarial signal misalignment**: The discriminators provide feedback on 16kHz audio structure, but the decoder must produce 24kHz audio.

### Concrete Example

For MS-STFT with `n_fft=1024, hop_length=256`:
- At 24kHz: Window = 42.7ms, Hop = 10.7ms, Max freq = 12kHz
- At 16kHz: Window = 64ms, Hop = 16ms, Max freq = 8kHz

The discriminator trained on 16kHz cannot guide 8-12kHz generation.

## Issue 3: Validation Also Affected

### Location
- `training/train.py` lines 490-518

### Code Analysis

```python
@torch.no_grad()
def validate(self) -> Dict[str, float]:
    # ...
    audio_recon = gen_module.decode_code(fsq_codes)
    
    # Same resampling applied in validation
    audio_recon = self.output_resampler.to(audio_recon.device)(audio_recon)
    
    mel_loss = self.mel_loss(audio_recon, audio)
```

### Problem
Validation metrics don't reflect true 24kHz quality, making it impossible to detect high-frequency degradation during training.

## Recommended Fixes

### Option A: Native 24kHz Loss Computation (Recommended)

```python
# Upsample input to 24kHz instead of downsampling output
class NeuCodecTrainer:
    def __init__(self, ...):
        # Upsample target to match output
        self.target_upsampler = torchaudio.transforms.Resample(16000, 24000)
        
        # Configure mel loss for 24kHz
        self.mel_loss = MultiResolutionMelLoss(
            sample_rate=24000,  # Native output rate
            n_ffts=[768, 1536, 3072],  # Scaled for 24kHz
            hop_lengths=[192, 384, 768],
            win_lengths=[768, 1536, 3072],
            n_mels=80,
        )
    
    def train_step(self, batch):
        audio = batch["audio"].to(self.device)  # 16kHz
        
        # Upsample target to 24kHz
        audio_24k = self.target_upsampler(audio)
        
        # Decode (output is 24kHz)
        audio_recon = gen_module.decode_code(fsq_codes)  # 24kHz
        
        # Compute loss at native 24kHz
        mel_loss = self.mel_loss(audio_recon, audio_24k)
```

### Option B: Dual-Rate Loss

```python
# Compute loss at both rates
mel_loss_16k = self.mel_loss_16k(audio_recon_16k, audio)
mel_loss_24k = self.mel_loss_24k(audio_recon, audio_24k)
total_mel_loss = mel_loss_16k + 0.5 * mel_loss_24k
```

### Option C: High-Frequency Specific Loss

```python
# Add explicit high-frequency supervision
def highfreq_loss(recon_24k, target_24k):
    # Bandpass filter 8-12kHz
    highfreq_recon = bandpass_filter(recon_24k, 8000, 12000, sr=24000)
    highfreq_target = bandpass_filter(target_24k, 8000, 12000, sr=24000)
    return F.l1_loss(highfreq_recon, highfreq_target)
```

## Verification Steps

To verify this issue is causing quality degradation:

1. **Spectral analysis**: Compare spectrograms of base vs finetuned outputs above 8kHz
2. **Frequency-band SNR**: Measure SNR separately for 0-4kHz, 4-8kHz, 8-12kHz bands
3. **Listening test**: Focus on sibilants, fricatives, and high-frequency transients

## References

- `training/train.py` - Main training loop
- `training/losses/mel_loss.py` - Mel spectrogram loss implementation
- `neucodec/model.py` - Model architecture showing 16kHz→24kHz design
- `training/configs/finetune_chatml.yaml` - Training configuration
