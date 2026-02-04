#!/usr/bin/env python3
"""
Test finetuned NeuCodec checkpoint - inspect and verify contents.
"""

import argparse
import torch
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")


def inspect_checkpoint(checkpoint_path: str):
    """Inspect checkpoint contents without loading full model."""
    print(f"Loading checkpoint: {checkpoint_path}")
    print(f"File size: {Path(checkpoint_path).stat().st_size / (1024**3):.2f} GB")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    print("\n" + "="*60)
    print("CHECKPOINT CONTENTS")
    print("="*60)
    
    for key in checkpoint.keys():
        if isinstance(checkpoint[key], torch.Tensor):
            print(f"  {key}: tensor {checkpoint[key].shape}")
        elif isinstance(checkpoint[key], dict):
            if key.endswith("_state_dict"):
                # Count parameters
                total_params = sum(v.numel() for v in checkpoint[key].values() if isinstance(v, torch.Tensor))
                print(f"  {key}: {len(checkpoint[key])} keys, {total_params/1e6:.1f}M params")
            else:
                print(f"  {key}: dict with {len(checkpoint[key])} keys")
        elif isinstance(checkpoint[key], (int, float)):
            print(f"  {key}: {checkpoint[key]}")
        else:
            print(f"  {key}: {type(checkpoint[key]).__name__}")
    
    # Print training info
    print("\n" + "="*60)
    print("TRAINING INFO")
    print("="*60)
    if "global_step" in checkpoint:
        print(f"  Training step: {checkpoint['global_step']}")
    if "epoch" in checkpoint:
        print(f"  Training epoch: {checkpoint['epoch']}")
    if "best_val_loss" in checkpoint:
        print(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")
    if "config" in checkpoint:
        print(f"  Config saved: Yes")
    
    # Verify required components
    print("\n" + "="*60)
    print("VERIFICATION")
    print("="*60)
    
    # Support both naming conventions
    has_generator = "generator_state_dict" in checkpoint or "generator" in checkpoint
    
    required_keys = [
        "global_step",
        "epoch",
    ]
    
    optional_keys = [
        "discriminators_state_dict",
        "discriminators",
        "optimizer_g_state_dict",
        "optimizer_g",
        "optimizer_d_state_dict",
        "optimizer_d",
        "scheduler_g_state_dict",
        "scheduler_g",
        "scheduler_d_state_dict",
        "scheduler_d",
        "best_val_loss",
        "config",
    ]
    
    # Check generator separately
    if has_generator:
        print(f"  [OK] generator weights")
    else:
        print(f"  [MISSING] generator weights")
        all_good = False
    
    all_good = True
    for key in required_keys:
        if key in checkpoint:
            print(f"  [OK] {key}")
        else:
            print(f"  [MISSING] {key}")
            all_good = False
    
    for key in optional_keys:
        if key in checkpoint:
            print(f"  [OK] {key}")
        else:
            print(f"  [--] {key} (optional)")
    
    # Check generator state dict structure
    gen_key = "generator_state_dict" if "generator_state_dict" in checkpoint else "generator"
    if gen_key in checkpoint:
        gen_state = checkpoint[gen_key]
        print("\n" + "="*60)
        print("GENERATOR STRUCTURE (sample keys)")
        print("="*60)
        keys = list(gen_state.keys())
        for key in keys[:15]:
            shape = gen_state[key].shape if isinstance(gen_state[key], torch.Tensor) else "N/A"
            print(f"  {key}: {shape}")
        if len(keys) > 15:
            print(f"  ... and {len(keys) - 15} more keys")
    
    print("\n" + "="*60)
    if all_good:
        print("STATUS: CHECKPOINT VALID ✓")
    else:
        print("STATUS: CHECKPOINT INCOMPLETE ✗")
    print("="*60)
    
    return checkpoint


def test_model_loading(checkpoint_path: str, device: str = "cpu"):
    """Test loading the model with checkpoint weights."""
    print("\n" + "="*60)
    print("TESTING MODEL LOADING")
    print("="*60)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Import and load model
    from neucodec import NeuCodec
    
    print("Loading base NeuCodec model...")
    model = NeuCodec.from_pretrained("neuphonic/neucodec")
    
    print("Loading finetuned weights...")
    # Support both naming conventions
    if "generator_state_dict" in checkpoint:
        state_dict = checkpoint["generator_state_dict"]
    elif "generator" in checkpoint:
        state_dict = checkpoint["generator"]
    else:
        print("ERROR: No generator weights found!")
        return None
    
    # Handle DDP wrapped keys (remove "module." prefix if present)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_state_dict[k[7:]] = v
        else:
            new_state_dict[k] = v
    
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print(f"  Missing keys: {len(missing)}")
    print(f"  Unexpected keys: {len(unexpected)}")
    if missing:
        print(f"  Sample missing: {missing[:5]}")
    if unexpected:
        print(f"  Sample unexpected: {unexpected[:5]}")
    
    model = model.to(device)
    model.eval()
    
    print("\nModel loaded successfully!")
    return model


def test_encode_decode(model, device: str = "cpu"):
    """Test encoding and decoding with dummy audio."""
    print("\n" + "="*60)
    print("TESTING ENCODE/DECODE")
    print("="*60)
    
    # Create dummy audio (3 seconds at 16kHz)
    dummy_audio = torch.randn(1, 1, 48000).to(device)
    print(f"Input shape: {dummy_audio.shape}")
    
    with torch.no_grad():
        # Encode (use encode_code method)
        codes = model.encode_code(dummy_audio)
        print(f"Encoded tokens shape: {codes.shape}")
        print(f"Token values range: [{codes.min().item()}, {codes.max().item()}]")
        
        # Decode (use decode_code method)
        reconstructed = model.decode_code(codes)
        print(f"Reconstructed shape: {reconstructed.shape}")
        print(f"Output range: [{reconstructed.min().item():.3f}, {reconstructed.max().item():.3f}]")
    
    print("\nEncode/Decode test PASSED ✓")


def main():
    parser = argparse.ArgumentParser(description="Test NeuCodec checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--test-loading", action="store_true", help="Test model loading")
    parser.add_argument("--test-inference", action="store_true", help="Test encode/decode")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    args = parser.parse_args()
    
    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"ERROR: Checkpoint not found: {args.checkpoint}")
        return
    
    # Inspect checkpoint
    checkpoint = inspect_checkpoint(args.checkpoint)
    
    # Test model loading if requested
    if args.test_loading or args.test_inference:
        model = test_model_loading(args.checkpoint, args.device)
        
        if args.test_inference:
            test_encode_decode(model, args.device)


if __name__ == "__main__":
    main()
