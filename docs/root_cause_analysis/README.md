# NeuCodec Fine-tuning Quality Degradation: Root Cause Analysis

**Date**: 2026-02-04  
**Status**: Analysis Complete  
**Severity**: High - Production quality impacted

---

## Executive Summary

Fine-tuning NeuCodec on Arabic TTS data produced a model with degraded audio quality compared to both the base model and original input audio. This analysis identifies **7 root causes** across 3 categories, with the primary issues being sample rate mismatches in the training pipeline.

### Key Findings

| Category | Issues Found | Severity |
|----------|-------------|----------|
| Sample Rate Handling | 3 | Critical |
| Decoder/ISTFT Head | 4 | High |
| Loss Configuration | 6 | Medium-High |

### Primary Root Causes

1. **Loss computed at 16kHz for 24kHz output** - No supervision for 8-12kHz frequencies
2. **ISTFT head magnitude bias drift** - Exponential mapping amplifies small weight changes
3. **Discriminators operate on wrong sample rate** - Adversarial signal misaligned

---

## Architecture Context

```
NeuCodec Pipeline:
┌──────────────┐     ┌───────────────┐     ┌──────────────┐     ┌──────────────┐
│ Input 16kHz  │────▶│ Encoder +     │────▶│ FSQ Quantizer│────▶│ Decoder +    │────▶ Output 24kHz
│              │     │ Semantic Enc  │     │ (50 tok/sec) │     │ ISTFT Head   │
└──────────────┘     └───────────────┘     └──────────────┘     └──────────────┘
```

Key architectural facts:
- Input sample rate: 16kHz
- Output sample rate: 24kHz (1.5x upsampling)
- Quantization: FSQ with 8 levels of 4 values each (65,536 codes)
- Decoder: Vocos backbone + ISTFT head

---

## Root Cause Details

### Category 1: Sample Rate Mismatch (Critical)

#### RC-1.1: Loss Computation on Resampled Audio

**File**: `training/train.py` lines 113-114, 398-405

```python
self.output_resampler = torchaudio.transforms.Resample(24000, 16000)
# ...
audio_recon = gen_module.decode_code(fsq_codes)  # 24kHz
audio_recon = self.output_resampler(audio_recon)  # Downsample to 16kHz
mel_loss = self.mel_loss(audio_recon, audio)  # Loss at 16kHz
```

**Impact**: 
- Frequencies 8-12kHz receive zero gradient
- Model produces artifacts in unsupervised frequency range
- Muffled/degraded high-frequency content

#### RC-1.2: Discriminator Sample Rate Mismatch

**File**: `training/train.py` lines 414-439

Discriminators (MPD, MSD, MS-STFT) configured for 16kHz but should supervise 24kHz content.

**Impact**:
- Adversarial loss doesn't guide high-frequency generation
- Perceptual quality degradation

#### RC-1.3: Mel Filterbank Configuration

**File**: `training/train.py` lines 346-353

Mel filterbank built with `sample_rate=16000`, limiting f_max to 8kHz.

**Impact**:
- 80 mel bands span only 0-8kHz
- No mel supervision for 8-12kHz

---

### Category 2: ISTFT Head / Decoder Issues (High)

#### RC-2.1: Exponential Magnitude Sensitivity

**File**: `neucodec/codec_decoder_vocos.py` lines 146-150

```python
mag = torch.exp(mag)  # Exponential mapping
```

Small bias changes cause exponential amplitude scaling:
- Bias shift of -0.3 → 26% amplitude reduction
- Bias shift of +0.3 → 35% amplitude increase

**Impact**: 
- Amplitude scaling drift during fine-tuning
- Volume inconsistency vs base model

#### RC-2.2: Phase Prediction Drift

Phase predictions can drift to large values, causing:
- Phase discontinuities
- Metallic/robotic artifacts
- Transient smearing

#### RC-2.3: Quantizer Codebook Drift

**File**: `neucodec/codec_decoder_vocos.py` lines 356-358

FSQ codebook is trainable; fine-tuning can shift embeddings causing encoder-decoder misalignment.

#### RC-2.4: Backbone Output Distribution Shift

Transformer backbone output statistics (mean, std) can shift, propagating through exponential ISTFT mapping.

---

### Category 3: Loss Configuration Issues (Medium-High)

#### RC-3.1: Commitment Loss Disabled

**File**: `training/configs/finetune_chatml.yaml` lines 143-145

```yaml
commitment:
  enabled: false
```

**Impact**: Quantizer instability, encoder drift from codebook.

#### RC-3.2: Loss Weight Imbalance

Current weights (mel:45, adv:1, fm:2) heavily favor mel reconstruction over adversarial guidance.

**Impact**: Over-smoothed outputs, weak perceptual quality signal.

#### RC-3.3: Missing Waveform-Level Loss

No direct waveform L1/L2 loss or amplitude matching loss.

**Impact**: Amplitude drift not penalized.

#### RC-3.4: Semantic Loss Not Applied

Semantic loss defined but not used in training loop.

**Impact**: Semantic content may degrade during fine-tuning.

#### RC-3.5: No Multi-Resolution STFT Loss

Only mel loss used; no spectral convergence + log magnitude loss.

**Impact**: Phase errors not directly supervised.

#### RC-3.6: Missing High-Frequency Supervision

No explicit loss for 8-12kHz band in 24kHz output.

---

## Severity Matrix

| Root Cause | Severity | Confidence | Fix Complexity |
|-----------|----------|------------|----------------|
| RC-1.1 Sample rate in loss | Critical | High | Medium |
| RC-1.2 Discriminator SR | Critical | High | Medium |
| RC-1.3 Mel filterbank | High | High | Low |
| RC-2.1 ISTFT magnitude | High | High | Low |
| RC-2.2 Phase drift | Medium | Medium | Medium |
| RC-2.3 Quantizer drift | Medium | Medium | Low |
| RC-2.4 Backbone shift | Medium | Medium | Low |
| RC-3.1 Commitment loss | Medium | High | Low |
| RC-3.2 Weight imbalance | Low | Medium | Low |
| RC-3.3 Waveform loss | Medium | Medium | Low |
| RC-3.4 Semantic loss | Low | High | Low |
| RC-3.5 STFT loss | Medium | Medium | Low |
| RC-3.6 HF supervision | High | High | Medium |

---

## Recommended Fixes (Priority Order)

### P0: Critical - Must Fix

1. **Compute losses at native 24kHz**
   ```python
   # Upsample target instead of downsampling output
   self.target_upsampler = torchaudio.transforms.Resample(16000, 24000)
   audio_24k = self.target_upsampler(audio)
   mel_loss = self.mel_loss_24k(audio_recon, audio_24k)
   ```

2. **Configure discriminators for 24kHz**
   ```yaml
   ms_stft:
     n_ffts: [1536, 3072, 768]  # Scaled 1.5x
     hop_lengths: [384, 768, 192]
   ```

3. **Update mel loss sample rate**
   ```python
   self.mel_loss = MultiResolutionMelLoss(
       sample_rate=24000,
       n_ffts=[768, 1536, 3072],
       hop_lengths=[192, 384, 768],
       n_mels=100,
       f_max=12000,
   )
   ```

### P1: High Priority

4. **Freeze or constrain ISTFT head**
   ```python
   for param in self.generator.generator.head.parameters():
       param.requires_grad = False
   ```

5. **Add amplitude preservation loss**
   ```python
   amp_loss = F.l1_loss(recon.abs().mean(), target.abs().mean())
   ```

6. **Enable commitment loss**
   ```yaml
   commitment:
     weight: 0.1
     enabled: true
   ```

### P2: Medium Priority

7. **Rebalance loss weights** (mel:30, adv:2, fm:5)
8. **Add multi-resolution STFT loss**
9. **Apply semantic loss to reconstruction**
10. **Lower learning rate for decoder components**

---

## Validation Plan

After implementing fixes:

1. **Spectral Analysis**
   - Compare 8-12kHz content in base vs new fine-tuned model
   - Measure high-frequency energy ratio

2. **Amplitude Verification**
   - Check amplitude statistics match base model
   - Verify consistent volume levels

3. **Listening Tests**
   - Focus on sibilants (s, sh, f sounds)
   - Check for metallic/robotic artifacts
   - Evaluate naturalness of Arabic phonemes

4. **Metrics**
   - PESQ/POLQA scores
   - MCD (Mel Cepstral Distortion)
   - High-frequency SNR (8-12kHz band)

---

## Detailed Documentation

For in-depth analysis of each category, see:

1. [Sample Rate Mismatch Analysis](01_sample_rate_mismatch.md)
2. [ISTFT Head and Decoder Analysis](02_istft_head_decoder.md)
3. [Loss Configuration Analysis](03_loss_configuration.md)

---

## Conclusion

The primary cause of quality degradation is the **sample rate mismatch** in the training pipeline. The model outputs 24kHz audio but all loss computation happens at 16kHz, leaving 33% of the frequency range (8-12kHz) completely unsupervised. This is compounded by the exponential sensitivity of the ISTFT head to weight changes.

Implementing the P0 fixes (native 24kHz loss computation) should resolve the majority of quality issues. P1 fixes (ISTFT constraints, amplitude loss) will address remaining amplitude and stability concerns.

---

## Files Modified/Created

- `docs/root_cause_analysis/README.md` (this file)
- `docs/root_cause_analysis/01_sample_rate_mismatch.md`
- `docs/root_cause_analysis/02_istft_head_decoder.md`
- `docs/root_cause_analysis/03_loss_configuration.md`

## Related Code Files

- `training/train.py` - Main training loop
- `training/configs/finetune_chatml.yaml` - Training configuration
- `training/losses/mel_loss.py` - Mel spectrogram loss
- `neucodec/codec_decoder_vocos.py` - Decoder and ISTFT head
- `neucodec/model.py` - Full model architecture
- `diagnose_decoder.py` - Weight analysis diagnostic tool
