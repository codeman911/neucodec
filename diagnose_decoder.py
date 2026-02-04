#!/usr/bin/env python3
"""Diagnose decoder weight changes between base and finetuned models."""

import torch
import numpy as np
from neucodec.model import NeuCodec
import warnings
warnings.filterwarnings('ignore')


def load_models():
    """Load base and finetuned models."""
    print("Loading base model from HuggingFace...")
    base_model = NeuCodec.from_pretrained("neuphonic/neucodec")
    base_model.eval()
    
    print("Loading finetuned checkpoint...")
    ft_model = NeuCodec.from_pretrained("neuphonic/neucodec")
    checkpoint = torch.load("checkpoints/finetune/checkpoint_best.pt", map_location='cpu')
    
    # Load generator weights
    gen_state = checkpoint['generator']
    missing, unexpected = ft_model.load_state_dict(gen_state, strict=False)
    print(f"Loaded checkpoint - missing: {len(missing)}, unexpected: {len(unexpected)}")
    ft_model.eval()
    
    return base_model, ft_model, checkpoint


def analyze_layer_weights(base_model, ft_model):
    """Compare weights layer by layer."""
    print("\n" + "="*80)
    print("DECODER WEIGHT ANALYSIS")
    print("="*80)
    
    # Key decoder components to analyze
    decoder_components = [
        ("fc_post_a", "Post-quantization projection (2048->1024)"),
        ("generator.backbone.embed", "Backbone embedding conv"),
        ("generator.backbone.final_layer_norm", "Final layer norm"),
        ("generator.head.out", "ISTFT output projection (1024->1282)"),
    ]
    
    results = []
    
    for component_name, description in decoder_components:
        print(f"\n{'-'*60}")
        print(f"Component: {component_name}")
        print(f"Description: {description}")
        
        # Get weight tensors
        try:
            base_params = dict(base_model.named_parameters())
            ft_params = dict(ft_model.named_parameters())
            
            weight_key = f"{component_name}.weight"
            bias_key = f"{component_name}.bias"
            
            if weight_key in base_params:
                base_w = base_params[weight_key].detach()
                ft_w = ft_params[weight_key].detach()
                
                # Compute statistics
                weight_diff = (ft_w - base_w).abs()
                relative_change = weight_diff / (base_w.abs() + 1e-8)
                
                print(f"\n  Weight shape: {base_w.shape}")
                print(f"  Base weight stats:")
                print(f"    mean={base_w.mean():.6f}, std={base_w.std():.6f}")
                print(f"    min={base_w.min():.6f}, max={base_w.max():.6f}")
                print(f"  Finetuned weight stats:")
                print(f"    mean={ft_w.mean():.6f}, std={ft_w.std():.6f}")
                print(f"    min={ft_w.min():.6f}, max={ft_w.max():.6f}")
                print(f"  Difference stats:")
                print(f"    mean_abs_diff={weight_diff.mean():.6f}")
                print(f"    max_abs_diff={weight_diff.max():.6f}")
                print(f"    mean_relative_change={relative_change.mean():.4f} ({relative_change.mean()*100:.2f}%)")
                
                # Check for scale changes
                scale_ratio = ft_w.std() / base_w.std()
                print(f"  Scale ratio (ft_std/base_std): {scale_ratio:.4f}")
                
                results.append({
                    'component': component_name,
                    'mean_diff': weight_diff.mean().item(),
                    'scale_ratio': scale_ratio.item(),
                    'relative_change': relative_change.mean().item()
                })
                
            if bias_key in base_params:
                base_b = base_params[bias_key].detach()
                ft_b = ft_params[bias_key].detach()
                bias_diff = (ft_b - base_b).abs()
                print(f"\n  Bias shape: {base_b.shape}")
                print(f"  Base bias: mean={base_b.mean():.6f}, std={base_b.std():.6f}")
                print(f"  Finetuned bias: mean={ft_b.mean():.6f}, std={ft_b.std():.6f}")
                print(f"  Bias diff: mean={bias_diff.mean():.6f}, max={bias_diff.max():.6f}")
                
        except Exception as e:
            print(f"  Error analyzing {component_name}: {e}")
    
    return results


def analyze_istft_head_detail(base_model, ft_model):
    """Deep analysis of ISTFT head - this directly affects audio amplitude."""
    print("\n" + "="*80)
    print("ISTFT HEAD DETAILED ANALYSIS (Critical for amplitude)")
    print("="*80)
    
    base_head = base_model.generator.head
    ft_head = ft_model.generator.head
    
    # The output linear layer determines magnitude and phase
    base_out_w = base_head.out.weight.detach()
    ft_out_w = ft_head.out.weight.detach()
    base_out_b = base_head.out.bias.detach()
    ft_out_b = ft_head.out.bias.detach()
    
    # Split into magnitude and phase parts (first half = mag, second half = phase)
    n_fft_half = base_out_w.shape[0] // 2
    
    base_mag_w = base_out_w[:n_fft_half]
    ft_mag_w = ft_out_w[:n_fft_half]
    base_phase_w = base_out_w[n_fft_half:]
    ft_phase_w = ft_out_w[n_fft_half:]
    
    base_mag_b = base_out_b[:n_fft_half]
    ft_mag_b = ft_out_b[:n_fft_half]
    base_phase_b = base_out_b[n_fft_half:]
    ft_phase_b = ft_out_b[n_fft_half:]
    
    print("\nMagnitude weights (controls amplitude):")
    print(f"  Base: mean={base_mag_w.mean():.6f}, std={base_mag_w.std():.6f}")
    print(f"  Finetuned: mean={ft_mag_w.mean():.6f}, std={ft_mag_w.std():.6f}")
    print(f"  Diff mean: {(ft_mag_w - base_mag_w).abs().mean():.6f}")
    
    print("\nMagnitude biases (affects base amplitude level):")
    print(f"  Base: mean={base_mag_b.mean():.6f}, std={base_mag_b.std():.6f}")
    print(f"  Finetuned: mean={ft_mag_b.mean():.6f}, std={ft_mag_b.std():.6f}")
    mag_bias_shift = ft_mag_b.mean() - base_mag_b.mean()
    print(f"  Mean bias shift: {mag_bias_shift:.6f}")
    
    # Since mag = exp(x), a negative bias shift means amplitude reduction
    print(f"\n  >>> Amplitude scale factor from bias shift: exp({mag_bias_shift:.4f}) = {np.exp(mag_bias_shift.item()):.4f}")
    
    print("\nPhase weights:")
    print(f"  Base: mean={base_phase_w.mean():.6f}, std={base_phase_w.std():.6f}")
    print(f"  Finetuned: mean={ft_phase_w.mean():.6f}, std={ft_phase_w.std():.6f}")
    
    print("\nPhase biases:")
    print(f"  Base: mean={base_phase_b.mean():.6f}, std={base_phase_b.std():.6f}")
    print(f"  Finetuned: mean={ft_phase_b.mean():.6f}, std={ft_phase_b.std():.6f}")


def analyze_backbone_transformers(base_model, ft_model):
    """Analyze transformer backbone changes."""
    print("\n" + "="*80)
    print("TRANSFORMER BACKBONE ANALYSIS")
    print("="*80)
    
    # Check each transformer block
    base_transformers = base_model.generator.backbone.transformers
    ft_transformers = ft_model.generator.backbone.transformers
    
    total_diff = 0
    total_params = 0
    
    for i, (base_block, ft_block) in enumerate(zip(base_transformers, ft_transformers)):
        block_diff = 0
        block_params = 0
        for (name, base_p), (_, ft_p) in zip(base_block.named_parameters(), ft_block.named_parameters()):
            diff = (ft_p - base_p).abs().mean().item()
            block_diff += diff * base_p.numel()
            block_params += base_p.numel()
        
        avg_diff = block_diff / block_params
        total_diff += block_diff
        total_params += block_params
        print(f"Transformer block {i}: avg_weight_diff = {avg_diff:.6f}")
    
    print(f"\nOverall transformer avg diff: {total_diff/total_params:.6f}")


def analyze_quantizer(base_model, ft_model):
    """Check if quantizer weights changed."""
    print("\n" + "="*80)
    print("QUANTIZER ANALYSIS")
    print("="*80)
    
    base_q = base_model.generator.quantizer
    ft_q = ft_model.generator.quantizer
    
    for (name, base_p), (_, ft_p) in zip(base_q.named_parameters(), ft_q.named_parameters()):
        diff = (ft_p - base_p).abs().mean().item()
        max_diff = (ft_p - base_p).abs().max().item()
        print(f"{name}: shape={base_p.shape}, mean_diff={diff:.6f}, max_diff={max_diff:.6f}")


def test_forward_pass(base_model, ft_model):
    """Test intermediate activations during forward pass."""
    print("\n" + "="*80)
    print("FORWARD PASS ACTIVATION ANALYSIS")
    print("="*80)
    
    # Create dummy FSQ codes (typical range for FSQ with levels [4,4,4,4,4,4,4,4])
    dummy_codes = torch.randint(0, 4, (1, 1, 100))  # [B, 1, T]
    
    with torch.no_grad():
        # Get quantizer embeddings
        base_emb = base_model.generator.quantizer.get_output_from_indices(dummy_codes.transpose(1, 2))
        ft_emb = ft_model.generator.quantizer.get_output_from_indices(dummy_codes.transpose(1, 2))
        
        print("\nAfter quantizer.get_output_from_indices:")
        print(f"  Base emb: shape={base_emb.shape}, mean={base_emb.mean():.4f}, std={base_emb.std():.4f}")
        print(f"  FT emb: shape={ft_emb.shape}, mean={ft_emb.mean():.4f}, std={ft_emb.std():.4f}")
        print(f"  Diff: {(ft_emb - base_emb).abs().mean():.6f}")
        
        # Apply fc_post_a
        base_post = base_model.fc_post_a(base_emb.transpose(1, 2)).transpose(1, 2)
        ft_post = ft_model.fc_post_a(ft_emb.transpose(1, 2)).transpose(1, 2)
        
        print("\nAfter fc_post_a:")
        print(f"  Base: mean={base_post.mean():.4f}, std={base_post.std():.4f}")
        print(f"  FT: mean={ft_post.mean():.4f}, std={ft_post.std():.4f}")
        print(f"  Diff: {(ft_post - base_post).abs().mean():.6f}")
        
        # Through backbone
        base_backbone = base_model.generator.backbone(base_post.transpose(1, 2))
        ft_backbone = ft_model.generator.backbone(ft_post.transpose(1, 2))
        
        print("\nAfter backbone:")
        print(f"  Base: mean={base_backbone.mean():.4f}, std={base_backbone.std():.4f}")
        print(f"  FT: mean={ft_backbone.mean():.4f}, std={ft_backbone.std():.4f}")
        print(f"  Diff: {(ft_backbone - base_backbone).abs().mean():.6f}")
        
        # Through ISTFT head
        base_audio, base_pred = base_model.generator.head(base_backbone)
        ft_audio, ft_pred = ft_model.generator.head(ft_backbone)
        
        print("\nHead intermediate (mag+phase prediction):")
        print(f"  Base pred: mean={base_pred.mean():.4f}, std={base_pred.std():.4f}")
        print(f"  FT pred: mean={ft_pred.mean():.4f}, std={ft_pred.std():.4f}")
        
        # Split mag and phase
        base_mag, base_phase = base_pred.chunk(2, dim=1)
        ft_mag, ft_phase = ft_pred.chunk(2, dim=1)
        
        print("\nMagnitude (before exp):")
        print(f"  Base: mean={base_mag.mean():.4f}, std={base_mag.std():.4f}, range=[{base_mag.min():.2f}, {base_mag.max():.2f}]")
        print(f"  FT: mean={ft_mag.mean():.4f}, std={ft_mag.std():.4f}, range=[{ft_mag.min():.2f}, {ft_mag.max():.2f}]")
        print(f"  >>> Mean shift: {ft_mag.mean() - base_mag.mean():.4f}")
        
        # Exp of magnitude
        base_mag_exp = torch.exp(base_mag).clamp(max=1e2)
        ft_mag_exp = torch.exp(ft_mag).clamp(max=1e2)
        print("\nMagnitude (after exp):")
        print(f"  Base: mean={base_mag_exp.mean():.4f}, std={base_mag_exp.std():.4f}")
        print(f"  FT: mean={ft_mag_exp.mean():.4f}, std={ft_mag_exp.std():.4f}")
        print(f"  >>> Ratio (ft/base): {ft_mag_exp.mean() / base_mag_exp.mean():.4f}")
        
        print("\nFinal audio output:")
        print(f"  Base: mean={base_audio.mean():.6f}, std={base_audio.std():.6f}, range=[{base_audio.min():.4f}, {base_audio.max():.4f}]")
        print(f"  FT: mean={ft_audio.mean():.6f}, std={ft_audio.std():.6f}, range=[{ft_audio.min():.4f}, {ft_audio.max():.4f}]")
        print(f"  >>> STD ratio (ft/base): {ft_audio.std() / base_audio.std():.4f}")


def check_training_loss_config(checkpoint):
    """Check training configuration from checkpoint."""
    print("\n" + "="*80)
    print("TRAINING CONFIGURATION FROM CHECKPOINT")
    print("="*80)
    
    if 'config' in checkpoint:
        config = checkpoint['config']
        print(f"\nLoss weights:")
        print(f"  mel_weight: {config.get('mel_weight', 'N/A')}")
        print(f"  adversarial_weight: {config.get('adversarial_weight', 'N/A')}")
        print(f"  feature_matching_weight: {config.get('feature_matching_weight', 'N/A')}")
        print(f"  semantic_weight: {config.get('semantic_weight', 'N/A')}")
        
        print(f"\nTraining info:")
        print(f"  global_step: {checkpoint.get('global_step', 'N/A')}")
        print(f"  best_val_loss: {checkpoint.get('best_val_loss', 'N/A')}")
        print(f"  learning_rate: {config.get('learning_rate', 'N/A')}")
    else:
        print("No config found in checkpoint")
        print(f"Available keys: {checkpoint.keys()}")


def main():
    base_model, ft_model, checkpoint = load_models()
    
    # Analyze training config
    check_training_loss_config(checkpoint)
    
    # Analyze decoder weights
    analyze_layer_weights(base_model, ft_model)
    
    # Deep dive into ISTFT head
    analyze_istft_head_detail(base_model, ft_model)
    
    # Check transformer backbone
    analyze_backbone_transformers(base_model, ft_model)
    
    # Check quantizer
    analyze_quantizer(base_model, ft_model)
    
    # Test forward pass
    test_forward_pass(base_model, ft_model)
    
    print("\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)
    print("""
Key things to check:
1. If magnitude bias shifted negative -> amplitude will be reduced (since mag = exp(x))
2. If backbone output has different scale -> affects all downstream computations
3. If quantizer changed -> embeddings will be different
4. Check 'Mean shift' in magnitude prediction - negative value = amplitude suppression
5. Check 'STD ratio' in final audio - should be close to 1.0 for similar quality
""")


if __name__ == "__main__":
    main()
