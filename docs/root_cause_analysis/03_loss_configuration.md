# Loss Configuration Analysis

## Overview

The loss configuration in NeuCodec fine-tuning combines multiple objectives: mel spectrogram reconstruction, adversarial losses, feature matching, and semantic preservation. Issues in loss weighting, configuration, and computation contribute to quality degradation.

## Current Loss Configuration

### Location
`training/configs/finetune_chatml.yaml` lines 123-145

```yaml
losses:
  mel:
    weight: 45.0
    n_ffts: [512, 1024, 2048]
    hop_lengths: [128, 256, 512]
    win_lengths: [512, 1024, 2048]
    n_mels: 80
    
  adversarial:
    weight: 1.0
    loss_type: "hinge"
    
  feature_matching:
    weight: 2.0
    
  semantic:
    weight: 1.5
    loss_type: "mse"
    
  commitment:
    weight: 0.25
    enabled: false  # DISABLED
```

## Issue 1: Mel Loss Sample Rate Mismatch

### Location
`training/train.py` lines 341-353

```python
def _build_losses(self):
    mel_config = loss_config["mel"]
    self.mel_loss = MultiResolutionMelLoss(
        sample_rate=self.config["data"]["sample_rate"],  # 16000 Hz
        n_ffts=mel_config["n_ffts"],      # [512, 1024, 2048]
        hop_lengths=mel_config["hop_lengths"],  # [128, 256, 512]
        win_lengths=mel_config["win_lengths"],  # [512, 1024, 2048]
        n_mels=mel_config["n_mels"],  # 80
    )
```

### Problem Description

The mel filterbank is constructed for 16kHz:

```python
# mel_loss.py lines 74-76
f_max = f_max or sample_rate // 2  # 8000 Hz for 16kHz
m_max = hz_to_mel(torch.tensor(f_max))  # ~2840 mels
```

This means:
- Maximum frequency captured: 8kHz
- 80 mel bands span 0-8kHz
- No supervision for 8-12kHz (present in 24kHz output)

### Frequency Coverage Analysis

| FFT Size | Freq Resolution | Max Freq (16kHz) | Max Freq (24kHz) |
|----------|-----------------|------------------|------------------|
| 512 | 31.25 Hz | 8 kHz | 12 kHz |
| 1024 | 15.63 Hz | 8 kHz | 12 kHz |
| 2048 | 7.81 Hz | 8 kHz | 12 kHz |

At 16kHz sample rate, all FFT configurations top out at 8kHz regardless of size.

## Issue 2: Discriminator Loss Operating on Wrong Sample Rate

### Location
`training/train.py` lines 414-439

```python
# Adversarial losses (after warmup)
if use_disc:
    disc_module = self.discriminators.module if self.distributed else self.discriminators
    
    for name, disc in disc_module.items():
        fake_out, fake_feat = disc(audio_recon)  # 16kHz resampled
        with torch.no_grad():
            real_out, real_feat = disc(audio)  # 16kHz original
```

### Problem Description

Discriminators see 16kHz audio but the model outputs 24kHz:
- MS-STFT discriminator cannot detect high-frequency artifacts
- MPD period patterns correspond to wrong frequencies
- MSD downsampling ratios misaligned

### MS-STFT Configuration Issue

```yaml
ms_stft:
  n_ffts: [1024, 2048, 512]
  hop_lengths: [256, 512, 128]
  win_lengths: [1024, 2048, 512]
```

For 16kHz audio with n_fft=2048:
- Frequency bins: 1025
- Max frequency: 8kHz
- Time resolution: 32ms per hop (at 512 hop)

For proper 24kHz supervision:
- Should use scaled parameters
- n_fft=3072 for equivalent resolution
- hop_length=768 for same temporal coverage

## Issue 3: Commitment Loss Disabled

### Location
`training/configs/finetune_chatml.yaml` lines 143-145

```yaml
commitment:
  weight: 0.25
  enabled: false
```

### Problem Description

Commitment loss from VQ-VAE helps:
1. Keep encoder outputs close to quantizer centroids
2. Prevent codebook collapse
3. Maintain encoder-decoder alignment

Without it during fine-tuning:
- Encoder can drift from codebook
- Same FSQ codes may decode differently
- Quality degradation in unseen inputs

### Expected Impact

The FSQ quantizer in NeuCodec uses 8 levels with 4 values each:
```python
self.quantizer = ResidualFSQ(
    dim=vq_dim, levels=[4, 4, 4, 4, 4, 4, 4, 4], num_quantizers=1
)
```

Total codebook: 4^8 = 65,536 possible codes

Without commitment loss, the embedding space can shift, causing:
- Inconsistent reconstruction
- Artifacts on edge cases
- Reduced generalization

## Issue 4: Loss Weight Imbalance

### Current Weights

| Loss | Weight | Effective Contribution |
|------|--------|----------------------|
| Mel | 45.0 | Dominant |
| Adversarial | 1.0 | Small |
| Feature Matching | 2.0 | Small |
| Semantic | 1.5 | Small |

### Problem Description

The mel loss is 45x stronger than adversarial loss. This means:
- Model heavily optimizes for mel reconstruction
- Adversarial signal (which improves perceptual quality) is weak
- May cause over-smoothed outputs

### Comparison to Common Practices

Typical GAN-based vocoders use:
- Mel weight: 15-45
- Adversarial weight: 1-4
- Feature matching weight: 2-10

The current ratio (45:1) is on the aggressive side for mel dominance.

## Issue 5: No Waveform-Level Loss

### Missing Component

Many high-quality neural vocoders include:
- L1/L2 waveform loss
- Multi-resolution STFT loss (amplitude + phase)
- Amplitude envelope loss

### Current Implementation Gap

```python
# train.py train_step - only mel loss for reconstruction
mel_loss = self.mel_loss(audio_recon, audio)
gen_loss = self.loss_weights["mel"] * mel_loss
```

No direct waveform supervision means:
- Phase errors not penalized directly
- Amplitude scaling can drift
- Temporal alignment issues possible

## Issue 6: Semantic Loss Not Applied to Reconstruction

### Location
`training/train.py` - semantic loss defined but usage unclear

```python
# Lines 363-365
self.semantic_loss = SemanticReconstructionLoss(
    loss_type=sem_config["loss_type"]
)
```

### Problem Description

The semantic loss is defined but not applied in `train_step()`. Examining the training loop:

```python
# train_step only uses:
# 1. mel_loss
# 2. adversarial loss
# 3. feature matching loss
# Semantic loss is NOT applied
```

This means the model has no constraint to preserve semantic content during fine-tuning.

## Recommended Fixes

### Fix 1: 24kHz Mel Loss Configuration

```yaml
losses:
  mel:
    weight: 45.0
    sample_rate: 24000  # Match output rate
    n_ffts: [768, 1536, 3072]  # Scaled 1.5x for 24kHz
    hop_lengths: [192, 384, 768]
    win_lengths: [768, 1536, 3072]
    n_mels: 100  # More bins for wider range
    f_max: 12000  # Full 24kHz range
```

### Fix 2: Enable Commitment Loss

```yaml
commitment:
  weight: 0.1  # Lower weight for fine-tuning
  enabled: true
```

### Fix 3: Rebalance Loss Weights

```yaml
losses:
  mel:
    weight: 30.0  # Reduced from 45
  adversarial:
    weight: 2.0   # Increased from 1
  feature_matching:
    weight: 5.0   # Increased from 2
```

### Fix 4: Add Multi-Resolution STFT Loss

```python
# In _build_losses()
from training.losses.mel_loss import MultiResolutionSTFTLoss

self.stft_loss = MultiResolutionSTFTLoss(
    n_ffts=[768, 1536, 3072],
    hop_lengths=[192, 384, 768],
    win_lengths=[768, 1536, 3072],
)

# In train_step()
stft_loss = self.stft_loss(audio_recon, audio_24k)
gen_loss = gen_loss + 5.0 * stft_loss
```

### Fix 5: Add Amplitude Preservation Loss

```python
def amplitude_loss(recon, target):
    """Match amplitude statistics."""
    recon_amp = recon.abs().mean(dim=-1)
    target_amp = target.abs().mean(dim=-1)
    return F.l1_loss(recon_amp, target_amp)

# In train_step()
amp_loss = amplitude_loss(audio_recon, audio)
gen_loss = gen_loss + 1.0 * amp_loss
```

### Fix 6: Apply Semantic Loss

```python
# In train_step() after reconstruction
with torch.no_grad():
    # Get semantic features of original
    orig_semantic = gen_module.semantic_model(
        semantic_features
    ).hidden_states[16]

# Get semantic features of reconstruction
recon_semantic = gen_module.semantic_model(
    recon_semantic_features
).hidden_states[16]

semantic_loss = self.semantic_loss(recon_semantic, orig_semantic)
gen_loss = gen_loss + self.loss_weights["semantic"] * semantic_loss
```

## Updated Loss Configuration (Recommended)

```yaml
losses:
  mel:
    weight: 30.0
    sample_rate: 24000
    n_ffts: [768, 1536, 3072]
    hop_lengths: [192, 384, 768]
    win_lengths: [768, 1536, 3072]
    n_mels: 100
    f_max: 12000
    
  stft:
    weight: 5.0
    n_ffts: [768, 1536, 3072]
    hop_lengths: [192, 384, 768]
    
  adversarial:
    weight: 2.0
    loss_type: "hinge"
    
  feature_matching:
    weight: 5.0
    
  semantic:
    weight: 1.5
    loss_type: "mse"
    apply_to_reconstruction: true
    
  commitment:
    weight: 0.1
    enabled: true
    
  amplitude:
    weight: 1.0
    enabled: true
```

## Verification Steps

1. Log individual loss components during training
2. Monitor mel loss at multiple resolutions separately
3. Check if adversarial loss is contributing (should not be near zero)
4. Verify semantic similarity between input and reconstruction
5. Compare amplitude statistics of outputs vs targets

## References

- `training/configs/finetune_chatml.yaml` - Loss configuration
- `training/losses/mel_loss.py` - Mel spectrogram loss
- `training/losses/discriminator_loss.py` - Adversarial losses
- `training/losses/semantic_loss.py` - Semantic preservation
- `training/train.py` - Training loop implementation
