# ISTFT Head and Decoder Analysis

## Overview

The ISTFT (Inverse Short-Time Fourier Transform) head is the final stage of the NeuCodec decoder that converts learned representations back to audio waveforms. Changes to this component during fine-tuning directly affect output amplitude and phase coherence.

## Architecture Reference

### Decoder Pipeline

```
FSQ Codes → Quantizer Embedding → fc_post_a → Backbone → ISTFT Head → Audio
   [B,1,F]      [B,F,2048]        [B,F,1024]   [B,F,1024]  [B,1,T]
```

### ISTFT Head Structure

Location: `neucodec/codec_decoder_vocos.py` lines 112-161

```python
class ISTFTHead(FourierHead):
    def __init__(self, dim: int, n_fft: int, hop_length: int, padding: str = "same"):
        super().__init__()
        out_dim = n_fft + 2  # 1920 + 2 = 1922 for hop_length=480
        self.out = torch.nn.Linear(dim, out_dim)  # 1024 → 1922
        self.istft = ISTFT(
            n_fft=n_fft, hop_length=hop_length, win_length=n_fft, padding=padding
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_pred = self.out(x)  # Linear projection
        x_pred = x_pred.transpose(1, 2)
        mag, p = x_pred.chunk(2, dim=1)  # Split into magnitude and phase
        
        mag = torch.exp(mag)  # EXPONENTIAL MAPPING
        mag = torch.clip(mag, max=1e2)  # Safety clamp
        
        # Phase wrapping
        x = torch.cos(p)
        y = torch.sin(p)
        S = mag * (x + 1j * y)  # Complex spectrogram
        
        audio = self.istft(S)
        return audio.unsqueeze(1), x_pred
```

## Issue 1: Exponential Magnitude Mapping Sensitivity

### Problem Description

The magnitude prediction uses exponential mapping: `mag = exp(x)`

This creates extreme sensitivity to weight/bias changes:

| Bias Shift | Amplitude Scale Factor |
|------------|----------------------|
| -0.5 | 0.607x (39% reduction) |
| -0.3 | 0.741x (26% reduction) |
| -0.1 | 0.905x (10% reduction) |
| +0.1 | 1.105x (10% increase) |
| +0.3 | 1.350x (35% increase) |
| +0.5 | 1.649x (65% increase) |

### Mathematical Analysis

For the output linear layer `self.out`:
- Weight shape: [1922, 1024]
- Bias shape: [1922]
- First 961 outputs control magnitude
- Last 961 outputs control phase

If fine-tuning shifts the magnitude bias by δ:
```
Original: mag = exp(Wx + b)
After:    mag = exp(Wx + b + δ) = exp(δ) × exp(Wx + b)
```

**A small bias drift causes multiplicative amplitude scaling across all frequencies.**

### Code Reference

```python
# codec_decoder_vocos.py lines 146-150
mag, p = x_pred.chunk(2, dim=1)
mag = torch.exp(mag)  # ← Exponential sensitivity here
mag = torch.clip(mag, max=1e2)
```

## Issue 2: Phase Prediction Drift

### Problem Description

Phase is predicted directly without wrapping constraints during forward pass:

```python
x = torch.cos(p)
y = torch.sin(p)
S = mag * (x + 1j * y)
```

While cos/sin provide natural wrapping, the phase prediction `p` can drift to arbitrary ranges during training. Large phase values can cause:
- Numerical instability in gradients
- Phase discontinuities between frames
- Artifacts in reconstructed audio

### Impact on Audio Quality

Phase errors manifest as:
- Metallic/robotic sound quality
- Pre-echo artifacts
- Loss of transient sharpness
- Reduced clarity in speech

## Issue 3: Backbone Output Distribution Shift

### Location
`neucodec/codec_decoder_vocos.py` lines 251-328

### Problem Description

The VocosBackbone contains:
- Embedding conv layer
- 2 ResNet blocks (prior_net)
- 12 Transformer blocks
- 2 ResNet blocks (post_net)
- Final LayerNorm

Fine-tuning can shift the output distribution of the backbone, which then feeds into the ISTFT head with its exponential mapping.

### Cascade Effect

```
Backbone output shift → ISTFT input shift → Exponential amplification → Amplitude distortion
```

A 10% shift in backbone output mean can cause 10-20% amplitude change due to exponential mapping.

## Issue 4: Quantizer-Decoder Alignment

### Location
`neucodec/codec_decoder_vocos.py` lines 356-358, 386-389

```python
# Quantizer
self.quantizer = ResidualFSQ(
    dim=vq_dim, levels=[4, 4, 4, 4, 4, 4, 4, 4], num_quantizers=1
)

# Embedding lookup during decode
def vq2emb(self, vq):
    self.quantizer = self.quantizer.eval()
    x = self.quantizer.vq2emb(vq)
    return x
```

### Problem Description

The FSQ quantizer produces embeddings that the decoder was trained to interpret. If fine-tuning modifies:
1. The quantizer codebook
2. The `fc_post_a` projection (2048→1024)

Then the embedding-to-audio mapping becomes misaligned.

## Diagnostic Analysis

### From diagnose_decoder.py

The diagnostic script analyzes key components:

```python
# Lines 105-154: ISTFT head analysis
def analyze_istft_head_detail(base_model, ft_model):
    # Split weights into magnitude and phase parts
    n_fft_half = base_out_w.shape[0] // 2
    
    base_mag_w = base_out_w[:n_fft_half]
    ft_mag_w = ft_out_w[:n_fft_half]
    
    # Calculate amplitude scale from bias shift
    mag_bias_shift = ft_mag_b.mean() - base_mag_b.mean()
    print(f"Amplitude scale factor: exp({mag_bias_shift}) = {np.exp(mag_bias_shift)}")
```

### Expected Diagnostic Output

If quality is degraded, expect to see:
- Magnitude bias shift (negative = amplitude reduction)
- Scale ratio ≠ 1.0 in forward pass analysis
- STD ratio in final audio ≠ 1.0

## Recommended Fixes

### Fix 1: Freeze ISTFT Head During Fine-tuning

```python
# In _build_models() after loading pretrained
def _build_models(self):
    self.generator = NeuCodec.from_pretrained("neuphonic/neucodec")
    
    # Freeze ISTFT head to preserve amplitude mapping
    for param in self.generator.generator.head.parameters():
        param.requires_grad = False
    logger.info("Froze ISTFT head parameters")
```

### Fix 2: Constrained Fine-tuning with Lower Learning Rate

```yaml
# finetune_chatml.yaml - Differential learning rates
optimizer:
  generator:
    type: "AdamW"
    lr: 5.0e-5
    param_groups:
      - params: "generator.head.*"
        lr: 1.0e-6  # 50x lower for ISTFT head
      - params: "generator.backbone.*"
        lr: 5.0e-5
```

### Fix 3: Amplitude Preservation Loss

```python
def amplitude_preservation_loss(recon, target):
    """Encourage similar amplitude statistics."""
    recon_rms = torch.sqrt(torch.mean(recon ** 2, dim=-1))
    target_rms = torch.sqrt(torch.mean(target ** 2, dim=-1))
    return F.l1_loss(recon_rms, target_rms)

# In train_step:
amp_loss = amplitude_preservation_loss(audio_recon, audio)
gen_loss = gen_loss + 0.5 * amp_loss
```

### Fix 4: Explicit Magnitude Bias Regularization

```python
def magnitude_bias_regularization(model, base_model):
    """Penalize drift from base model's magnitude biases."""
    head = model.generator.head
    base_head = base_model.generator.head
    
    n_fft_half = head.out.bias.shape[0] // 2
    mag_bias = head.out.bias[:n_fft_half]
    base_mag_bias = base_head.out.bias[:n_fft_half].detach()
    
    return F.mse_loss(mag_bias, base_mag_bias)
```

### Fix 5: Post-hoc Amplitude Correction

For already-trained models, apply correction factor:

```python
def correct_amplitude(finetuned_model, base_model, test_input):
    """Calculate and apply amplitude correction."""
    with torch.no_grad():
        base_out = base_model.decode_code(test_input)
        ft_out = finetuned_model.decode_code(test_input)
        
        # Calculate correction factor
        correction = base_out.std() / ft_out.std()
        
    # Apply during inference
    def corrected_decode(codes):
        audio = finetuned_model.decode_code(codes)
        return audio * correction
    
    return corrected_decode, correction
```

## Verification Checklist

- [ ] Run `diagnose_decoder.py` and check magnitude bias shift
- [ ] Compare backbone output distributions (mean, std)
- [ ] Measure amplitude ratio between base and finetuned outputs
- [ ] Check phase prediction statistics (should be bounded)
- [ ] Verify quantizer embeddings haven't drifted significantly

## References

- `neucodec/codec_decoder_vocos.py` - Decoder and ISTFT implementation
- `diagnose_decoder.py` - Diagnostic tool for weight analysis
- `neucodec/model.py` - Full model architecture
