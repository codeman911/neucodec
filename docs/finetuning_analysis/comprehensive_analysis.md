# NeuCodec Fine-tuning Comprehensive Analysis

**Date**: 2026-02-04  
**Version**: 1.0  
**Status**: Complete Analysis

---

## Table of Contents

1. [Technical Changes During Fine-tuning](#1-technical-changes-during-fine-tuning)
2. [Performance Comparison Analysis](#2-performance-comparison-analysis)
3. [Critical Evaluation of Current Approach](#3-critical-evaluation-of-current-approach)
4. [Research and Recommendations](#4-research-and-recommendations)
5. [Risk Assessment](#5-risk-assessment)
6. [Executive Summary and Action Plan](#6-executive-summary-and-action-plan)

---

## 1. Technical Changes During Fine-tuning

### 1.1 Architectural Configuration

#### Base Model Architecture (from `neucodec/model.py`)

| Component | Configuration | Parameters |
|-----------|--------------|------------|
| **Semantic Encoder** | Wav2Vec2-BERT 2.0, Layer 16 | ~580M (frozen) |
| **Acoustic Encoder** | CodecEncoder (BigCodec-based) | ~15M |
| **FSQ Quantizer** | 8 levels × 4 values, dim=2048 | ~8M |
| **Decoder** | Vocos Backbone (12 layers) | ~85M |
| **ISTFT Head** | n_fft=1920, hop=480 | ~2.5M |
| **Total Trainable** | (excluding semantic) | ~110M |

```
Input 16kHz → Encoder → FSQ Codes (50 tok/s) → Decoder → Output 24kHz
     ↓
Wav2Vec2-BERT → Semantic Features → Concatenate → fc_prior (2048→2048)
```

#### Fine-tuning Modifications

| Aspect | Base Training | Fine-tuning Change |
|--------|--------------|-------------------|
| Semantic Encoder | Frozen | Frozen (unchanged) |
| Acoustic Encoder | Trainable | Trainable |
| FSQ Quantizer | Trainable | Trainable |
| Decoder Backbone | Trainable | Trainable |
| ISTFT Head | Trainable | Trainable |
| fc_post_a | Trainable | Trainable |

**Key Observation**: No architectural modifications were made during fine-tuning. All components remain identical to the base model structure.

### 1.2 Loss Function Configuration

#### Base Training Configuration (`neucodec_train.yaml`)

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
    weight: 1.0
    loss_type: "mse"
  commitment:
    weight: 0.25
    enabled: false
```

#### Fine-tuning Configuration (`finetune_chatml.yaml`)

```yaml
losses:
  mel:
    weight: 45.0          # Same
    n_ffts: [512, 1024, 2048]
    hop_lengths: [128, 256, 512]
    win_lengths: [512, 1024, 2048]
    n_mels: 80
  adversarial:
    weight: 1.0           # Same
    loss_type: "hinge"
  feature_matching:
    weight: 2.0           # Same
  semantic:
    weight: 1.5           # Increased from 1.0 to 1.5
    loss_type: "mse"
  commitment:
    weight: 0.25
    enabled: false        # Still disabled
```

**Key Changes**:
- Semantic loss weight increased 50% (1.0 → 1.5)
- All other loss weights unchanged
- Commitment loss remains disabled

### 1.3 Training Hyperparameters

| Parameter | Base Training | Fine-tuning | Change |
|-----------|--------------|-------------|--------|
| Total Steps | 1,000,000 | 200,000 | -80% |
| Warmup Steps | 10,000 | 5,000 | -50% |
| Learning Rate (G) | 1.0e-4 | 5.0e-5 | -50% |
| Learning Rate (D) | 1.0e-4 | 5.0e-5 | -50% |
| Batch Size | 32 | 16 | -50% |
| Segment Length | 32,000 (2s) | 48,000 (3s) | +50% |
| LR Scheduler Gamma | 0.999875 | 0.99995 | Slower decay |
| Gradient Accumulation | 4 | 4 | Same |
| Discriminator Start | 0 | 0 | Same |

### 1.4 Dataset Characteristics

#### Training Data (ChatML Format)
- **Sources**: Gulf TTS v3, TTS Dialect v1
- **Languages**: Arabic (Gulf dialects: Emirati, Saudi)
- **Duration Range**: 0.5s - 30s
- **Validation Split**: 5% auto-split from training data
- **Sample Rate**: 16kHz input

#### Augmentation (Reduced for Fine-tuning)

| Augmentation | Base | Fine-tuning | Rationale |
|-------------|------|-------------|-----------|
| Noise probability | 0.3 | 0.1 | Preserve voice quality |
| Noise SNR range | [5, 30] | [15, 35] | Less aggressive |
| Reverb probability | 0.2 | 0.05 | CPU intensive |
| Pitch shift | 0.1 | 0.0 | Disabled |
| Time stretch | 0.1 | 0.0 | Disabled |

### 1.5 Preprocessing Pipeline

```python
# From training/data/dataloader.py
def _normalize(self, waveform: torch.Tensor) -> torch.Tensor:
    """Normalize waveform to [-1, 1]."""
    max_val = waveform.abs().max()
    if max_val > 0:
        waveform = waveform / max_val * 0.95  # Peak at 0.95
    return waveform
```

**Critical Issue**: Peak normalization to 0.95 may differ from base model's training distribution.

---

## 2. Performance Comparison Analysis

### 2.1 Quantitative Metrics (from `comparison_results.json`)

#### NeuCodec V1 Base Model Results

| File | SNR (dB) | PSNR (dB) | RMSE | Correlation | RTF |
|------|----------|-----------|------|-------------|-----|
| emirati_female_1 | -1.67 | 25.49 | 0.0414 | 0.275 | 0.050 |
| emirati_male_1 | -0.06 | 26.82 | 0.0324 | 0.464 | 0.010 |
| saudi_female_1 | -0.19 | 20.64 | 0.0758 | 0.427 | 0.011 |
| saudi_male_1 | +0.80 | 21.20 | 0.0545 | 0.592 | 0.010 |
| **Average** | **-0.28** | **23.54** | **0.0510** | **0.440** | **0.020** |

#### Expected Fine-tuned Model Metrics

Based on checkpoint metadata:
- **Training Step**: 32,500 (of 200,000 planned)
- **Best Validation Loss**: 0.414 (mel loss)
- **Checkpoint**: `checkpoint_best.pt`

### 2.2 Audio Quality Analysis

#### Sample Rate Handling

| Stage | Sample Rate | Notes |
|-------|-------------|-------|
| Input audio | 16 kHz | ChatML dataset |
| After encoding | 50 tokens/s | FSQ codes |
| After decoding | 24 kHz | 1.5x upsampling |
| Loss computation | 16 kHz | **MISMATCH** |
| Discriminator input | 16 kHz | **MISMATCH** |

**Critical Finding**: 
```python
# train.py line 114 - Resampler for loss computation
self.output_resampler = torchaudio.transforms.Resample(24000, 16000)

# train.py lines 398-400 - Output is downsampled for loss
audio_recon = gen_module.decode_code(fsq_codes)  # 24kHz
audio_recon = self.output_resampler(audio_recon)  # → 16kHz
mel_loss = self.mel_loss(audio_recon, audio)  # Loss at 16kHz
```

This means **8-12kHz content receives zero gradient supervision**.

### 2.3 Computational Efficiency

| Metric | Base Model | After Fine-tuning |
|--------|------------|-------------------|
| Encode RTF | ~0.008 | ~0.008 (no change) |
| Decode RTF | ~0.003 | ~0.003 (no change) |
| Total RTF | ~0.011 | ~0.011 (no change) |
| Parameters | 187.6M | 187.6M (no change) |

**Note**: Fine-tuning does not affect computational efficiency as architecture is unchanged.

### 2.4 Token Difference Analysis

From `compare_models_v2.py` output format:
- Token difference measures FSQ code changes between base and fine-tuned model
- Higher token difference indicates more divergence from base behavior
- Low token difference (~0%) suggests fine-tuning had minimal encoder impact

---

## 3. Critical Evaluation of Current Approach

### 3.1 Identified Weaknesses

#### Issue 1: Sample Rate Mismatch (CRITICAL)

**Severity**: Critical  
**Impact**: 33% of frequency range unsupervised

```
Problem:
┌─────────┐    ┌─────────┐    ┌─────────┐
│ 16kHz   │───▶│ Encoder │───▶│ Decoder │───▶ 24kHz output
│ input   │    │         │    │         │    
└─────────┘    └─────────┘    └─────────┘
                                    │
                              Resample to 16kHz ← INFORMATION LOSS
                                    │
                                    ▼
                              Loss @ 16kHz ← NO HF GRADIENTS
```

**Evidence**: Mel filterbank in loss function:
- `f_max = sample_rate // 2 = 8000 Hz`
- 80 mel bands span 0-8kHz only
- 8-12kHz in output gets no supervision

#### Issue 2: ISTFT Head Amplitude Sensitivity (HIGH)

**Severity**: High  
**Impact**: Exponential amplitude drift

```python
# codec_decoder_vocos.py lines 146-150
mag = torch.exp(mag)  # EXPONENTIAL MAPPING
```

Mathematical impact of bias drift δ:
```
new_amplitude = exp(δ) × original_amplitude

δ = -0.3  →  0.74× amplitude (26% reduction)
δ = +0.3  →  1.35× amplitude (35% increase)
```

#### Issue 3: Discriminator Configuration Mismatch (HIGH)

**Severity**: High  
**Impact**: Adversarial signal misaligned

MS-STFT discriminator configured for 16kHz:
```yaml
ms_stft:
  n_ffts: [1024, 2048, 512]
  hop_lengths: [256, 512, 128]  # For 16kHz
```

For 24kHz output, should be:
```yaml
ms_stft:
  n_ffts: [1536, 3072, 768]     # 1.5× scaled
  hop_lengths: [384, 768, 192]  # 1.5× scaled
```

#### Issue 4: Commitment Loss Disabled (MEDIUM)

**Severity**: Medium  
**Impact**: Quantizer drift, encoder-decoder misalignment

```yaml
commitment:
  weight: 0.25
  enabled: false  # DISABLED
```

Without commitment loss:
- FSQ codebook can drift during fine-tuning
- Same input may produce different codes
- Decoder receives shifted embeddings

#### Issue 5: Semantic Loss Not Applied (MEDIUM)

**Severity**: Medium  
**Impact**: Semantic content may degrade

The semantic loss is defined but not used in `train_step()`:
```python
# Defined in _build_losses()
self.semantic_loss = SemanticReconstructionLoss(loss_type="mse")

# NOT applied in train_step() - only mel, adversarial, feature_matching used
```

### 3.2 Why Previous Fine-tuning Produced Similar Results

Based on analysis:

1. **Identical Architecture**: No architectural changes that would improve quality
2. **Same Loss Functions**: Loss configuration nearly identical to base
3. **Sample Rate Bug**: Training optimizes for 16kHz but outputs 24kHz
4. **Limited Training**: Only 32,500 steps of 200,000 planned
5. **Conservative LR**: 50% lower learning rate limits adaptation

### 3.3 Root Causes Summary

| Root Cause | Category | Severity | Fix Complexity |
|-----------|----------|----------|----------------|
| 16kHz loss for 24kHz output | Sample Rate | Critical | Medium |
| ISTFT exp() sensitivity | Decoder | High | Low |
| Discriminator SR mismatch | Sample Rate | High | Medium |
| No HF supervision | Loss Config | High | Medium |
| Commitment loss disabled | Loss Config | Medium | Low |
| Semantic loss unused | Loss Config | Medium | Low |
| Normalization mismatch | Data Pipeline | Low | Low |

---

## 4. Research and Recommendations

### 4.1 Literature Review

#### Relevant Architectures

| Model | Key Innovation | Relevance |
|-------|---------------|-----------|
| **BigCodec** | Low-bitrate (1.04kbps) with VQ-GAN | NeuCodec is based on this |
| **Vocos** | ISTFT-based fourier vocoder | Same decoder approach |
| **DAC** | Multi-scale RVQ, improved discriminators | Alternative quantization |
| **SNAC** | Multi-scale neural codec | Hierarchical quantization |
| **EnCodec** | RVQ + streaming support | Meta's production codec |

#### Key Techniques from Literature

1. **Multi-Resolution STFT Loss** (UnivNet, Vocos)
   - Spectral convergence + log magnitude
   - Better phase supervision

2. **Discriminator Design** (HiFi-GAN, BigCodec)
   - MPD + MSD combination
   - Sample-rate-aware configuration

3. **Commitment Loss** (VQ-VAE, SoundStream)
   - Encoder-codebook alignment
   - Prevents codebook collapse

4. **Adversarial Training Balance** (StyleGAN, WaveGAN)
   - Feature matching helps stabilize training
   - R1 regularization for discriminator

### 4.2 Recommended Improvements

#### Fix 1: Native 24kHz Loss Computation (P0)

```python
# train.py - Replace output resampling with target upsampling
class NeuCodecTrainer:
    def __init__(self, ...):
        # Upsample target to match output rate
        self.target_upsampler = torchaudio.transforms.Resample(16000, 24000)
        
        # Configure mel loss for 24kHz
        self.mel_loss = MultiResolutionMelLoss(
            sample_rate=24000,
            n_ffts=[768, 1536, 3072],      # Scaled 1.5x
            hop_lengths=[192, 384, 768],    # Scaled 1.5x
            win_lengths=[768, 1536, 3072],
            n_mels=100,                     # More bins for wider range
            f_max=12000,                    # Full 24kHz range
        )
    
    def train_step(self, batch):
        audio = batch["audio"].to(self.device)  # 16kHz
        audio_24k = self.target_upsampler(audio)  # Upsample to 24kHz
        
        # ... encode/decode ...
        
        audio_recon = gen_module.decode_code(fsq_codes)  # 24kHz
        
        # Loss at native 24kHz - CORRECT
        mel_loss = self.mel_loss(audio_recon, audio_24k)
```

**Expected Impact**: Full frequency supervision, improved HF quality

#### Fix 2: 24kHz Discriminator Configuration (P0)

```yaml
# finetune_chatml.yaml
discriminators:
  mpd:
    enabled: true
    periods: [2, 3, 5, 7, 11, 13]  # Add 13 for 24kHz
    
  ms_stft:
    enabled: true
    filters: 32
    n_ffts: [1536, 3072, 768]      # Scaled for 24kHz
    hop_lengths: [384, 768, 192]
    win_lengths: [1536, 3072, 768]
```

**Expected Impact**: Proper adversarial guidance for full spectrum

#### Fix 3: Freeze ISTFT Head (P1)

```python
# _build_models() - Freeze ISTFT head to preserve base amplitude
for param in self.generator.generator.head.parameters():
    param.requires_grad = False
logger.info("Froze ISTFT head to preserve amplitude characteristics")
```

**Expected Impact**: Prevent amplitude drift, maintain base model's output scale

#### Fix 4: Add Amplitude Preservation Loss (P1)

```python
def amplitude_preservation_loss(recon, target):
    """Match amplitude statistics between reconstruction and target."""
    recon_rms = torch.sqrt(torch.mean(recon ** 2, dim=-1))
    target_rms = torch.sqrt(torch.mean(target ** 2, dim=-1))
    return F.l1_loss(recon_rms, target_rms)

# In train_step:
amp_loss = amplitude_preservation_loss(audio_recon, audio_24k)
gen_loss = gen_loss + 0.5 * amp_loss
```

**Expected Impact**: Consistent amplitude, better volume matching

#### Fix 5: Enable Commitment Loss (P2)

```yaml
commitment:
  weight: 0.1  # Lower for fine-tuning stability
  enabled: true
```

**Expected Impact**: Stable quantizer, better generalization

#### Fix 6: Add Multi-Resolution STFT Loss (P2)

```python
# Add to _build_losses()
self.stft_loss = MultiResolutionSTFTLoss(
    n_ffts=[768, 1536, 3072],
    hop_lengths=[192, 384, 768],
    win_lengths=[768, 1536, 3072],
)

# In train_step():
stft_loss = self.stft_loss(audio_recon, audio_24k)
gen_loss = gen_loss + 5.0 * stft_loss
```

**Expected Impact**: Better phase coherence, reduced artifacts

### 4.3 Proposed Updated Configuration

```yaml
# finetune_chatml_v2.yaml - Corrected Configuration

model:
  sample_rate: 16000
  output_sample_rate: 24000
  hop_length: 480

training:
  total_steps: 200000
  warmup_steps: 5000
  lr_generator: 5.0e-5
  lr_discriminator: 5.0e-5
  
  # New: Freeze ISTFT head
  freeze_istft_head: true
  
losses:
  mel:
    weight: 30.0          # Reduced from 45
    sample_rate: 24000    # CRITICAL: Match output rate
    n_ffts: [768, 1536, 3072]
    hop_lengths: [192, 384, 768]
    win_lengths: [768, 1536, 3072]
    n_mels: 100
    f_max: 12000
    
  stft:                   # NEW: Add STFT loss
    weight: 5.0
    n_ffts: [768, 1536, 3072]
    hop_lengths: [192, 384, 768]
    
  adversarial:
    weight: 2.0           # Increased from 1.0
    loss_type: "hinge"
    
  feature_matching:
    weight: 5.0           # Increased from 2.0
    
  amplitude:              # NEW: Amplitude preservation
    weight: 0.5
    enabled: true
    
  commitment:
    weight: 0.1
    enabled: true         # ENABLED
    
discriminators:
  ms_stft:
    enabled: true
    filters: 32
    n_ffts: [1536, 3072, 768]      # Scaled for 24kHz
    hop_lengths: [384, 768, 192]
    win_lengths: [1536, 3072, 768]
```

### 4.4 Future-Proof Approaches

#### Short-term (Next Training Run)
1. Implement P0 fixes (sample rate)
2. Implement P1 fixes (ISTFT, amplitude)
3. Retrain with corrected pipeline
4. Benchmark against base model

#### Medium-term (Architecture Improvements)
1. Explore hierarchical quantization (like SNAC)
2. Add streaming support for real-time inference
3. Implement multi-scale decoder for quality/efficiency tradeoff

#### Long-term (Market Competitiveness)
1. Support multiple sample rates (16/24/48kHz)
2. Variable bitrate encoding (0.5-6 kbps)
3. Semantic-guided generation for TTS integration
4. On-device optimization (quantization, pruning)

---

## 5. Risk Assessment

### 5.1 Current Approach Risks

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| **HF quality degradation** | High | High | Fix sample rate mismatch |
| **Amplitude inconsistency** | High | Medium | Freeze ISTFT head |
| **Quantizer drift** | Medium | Medium | Enable commitment loss |
| **Training instability** | Low | High | Lower LR, gradient clipping |
| **Overfitting to Arabic** | Medium | Low | Larger dataset, regularization |

### 5.2 Missing Elements

1. **No perceptual loss** (e.g., PESQ-based, deep feature matching)
2. **No speaker consistency loss** (for TTS applications)
3. **No explicit prosody preservation**
4. **No streaming-aware training**
5. **No multi-language evaluation**

### 5.3 Potential Failure Points

| Failure Point | Symptom | Detection | Prevention |
|--------------|---------|-----------|------------|
| HF artifacts | Metallic sound | Spectrogram analysis | 24kHz supervision |
| Amplitude drift | Quiet/loud output | RMS comparison | Amplitude loss |
| Phase errors | Pre-echo, smearing | Listening test | STFT loss |
| Codebook collapse | Repeated tokens | Token histogram | Commitment loss |
| Mode collapse | Monotone output | Diversity metrics | R1 regularization |

### 5.4 Confidence Levels

| Improvement | Confidence | Expected Gain |
|-------------|-----------|---------------|
| 24kHz loss computation | 90% | High (HF quality) |
| ISTFT head freezing | 85% | Medium (amplitude) |
| Discriminator fix | 80% | Medium (perceptual) |
| Commitment loss | 70% | Low-Medium (stability) |
| STFT loss addition | 75% | Medium (phase) |

---

## 6. Executive Summary and Action Plan

### Summary

The NeuCodec fine-tuning pipeline has a **critical architectural bug**: loss computation occurs at 16kHz while the model outputs 24kHz audio. This means 33% of the frequency spectrum (8-12kHz) receives no gradient supervision, leading to degraded high-frequency quality.

Secondary issues include:
- ISTFT head's exponential magnitude mapping causes amplitude sensitivity
- Discriminators configured for wrong sample rate
- Several beneficial losses disabled or unused

### Immediate Action Plan

#### Phase 1: Critical Fixes (Required)
- [ ] Modify `train.py` to upsample targets instead of downsampling outputs
- [ ] Update mel loss configuration for 24kHz
- [ ] Update discriminator STFT parameters for 24kHz
- [ ] Add option to freeze ISTFT head

#### Phase 2: Recommended Improvements
- [ ] Add amplitude preservation loss
- [ ] Enable commitment loss
- [ ] Add multi-resolution STFT loss
- [ ] Apply semantic loss to reconstruction

#### Phase 3: Validation
- [ ] Retrain model with corrected pipeline
- [ ] Compare spectrograms (base vs new) focusing on 8-12kHz
- [ ] Measure SNR, correlation, RMSE improvements
- [ ] Conduct listening tests for Arabic speech quality

### Expected Outcomes

With the recommended fixes:
- **HF Quality**: Significant improvement (8-12kHz supervision)
- **Amplitude**: Consistent with base model
- **Perceptual Quality**: Improved through better adversarial training
- **Stability**: Better generalization through commitment loss

### Resource Requirements

| Item | Estimate |
|------|----------|
| Code changes | ~200 lines |
| Configuration updates | ~50 lines |
| Testing/validation | ~2-4 GPU hours |
| Full retraining | ~24-48 GPU hours |

---

## References

### Internal Documentation
- `docs/root_cause_analysis/01_sample_rate_mismatch.md`
- `docs/root_cause_analysis/02_istft_head_decoder.md`
- `docs/root_cause_analysis/03_loss_configuration.md`

### External Literature
- BigCodec: Low-Bitrate Neural Speech Codec (arXiv:2409.05377)
- Vocos: Closing the Gap Between Time-Domain and Fourier Vocoders (arXiv:2306.00814)
- HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis
- SoundStream: An End-to-End Neural Audio Codec (Google, 2021)
- EnCodec: High Fidelity Neural Audio Compression (Meta, 2022)

### Code Files
- `training/train.py` - Main training loop
- `training/configs/finetune_chatml.yaml` - Fine-tuning configuration
- `training/losses/mel_loss.py` - Mel spectrogram loss
- `neucodec/codec_decoder_vocos.py` - Decoder and ISTFT head
- `neucodec/model.py` - Full model architecture
- `compare_models_v2.py` - Comparison evaluation script
- `diagnose_decoder.py` - Weight analysis diagnostic tool
