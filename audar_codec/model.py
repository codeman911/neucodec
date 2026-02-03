"""
Audar-Codec: Main Model Interface

Provides the complete Audar-Codec model with:
- Streaming encode/decode capabilities
- Backward compatibility with NeuCodec V1
- Hierarchical quantization support (Phase 3)
"""

from typing import Optional, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core.streaming_decoder import StreamingCodecDecoder, DecoderStreamingState
from .migration.v1_to_audar_loader import (
    load_v1_checkpoint,
    migrate_v1_to_audar,
    AudarMigrationConfig,
)


@dataclass
class AudarCodecConfig:
    """
    Configuration for Audar-Codec.
    
    Attributes:
        sample_rate: Audio sample rate (24000 Hz)
        hop_length: Hop length for STFT (480 samples = 50Hz at 24kHz)
        hidden_dim: Model hidden dimension (1024)
        depth: Number of transformer layers (12)
        heads: Number of attention heads (16)
        rope_dim: Rotary position embedding dimension (64)
        vq_dim: Vector quantization dimension (2048)
        vq_levels: FSQ levels per dimension ([4,4,4,4,4,4,4,4] = 8 dims)
        semantic_model: Semantic encoder model name
        streaming_chunk_size: Default chunk size for streaming (in frames)
    """
    # Audio settings
    sample_rate: int = 24000
    hop_length: int = 480  # 50Hz frame rate
    
    # Decoder architecture
    hidden_dim: int = 1024
    depth: int = 12
    heads: int = 16
    rope_dim: int = 64
    
    # Quantization
    vq_dim: int = 2048
    vq_levels: list = field(default_factory=lambda: [4, 4, 4, 4, 4, 4, 4, 4])
    
    # Semantic encoder
    semantic_model: str = "facebook/wav2vec2-xls-r-300m"
    semantic_dim: int = 1024  # XLS-R 300M output dim
    
    # Streaming
    streaming_chunk_size: int = 10  # frames
    max_seq_length: int = 8192  # max sequence for KV-cache
    
    # Phase settings
    enable_hierarchical: bool = False  # Phase 3 feature
    coarse_frame_rate: float = 12.5  # Hz, for LLM compatibility
    fine_frame_rate: float = 50.0  # Hz, for quality
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.sample_rate > 0
        assert self.hop_length > 0
        assert self.hidden_dim % self.heads == 0
    
    @classmethod
    def from_v1(cls) -> "AudarCodecConfig":
        """Create config matching NeuCodec V1."""
        return cls(
            sample_rate=24000,
            hop_length=480,
            hidden_dim=1024,
            depth=12,
            heads=16,
            rope_dim=64,
            vq_dim=2048,
            semantic_model="facebook/w2v-bert-2.0",
            semantic_dim=1024,
        )
    
    @classmethod
    def streaming_optimized(cls) -> "AudarCodecConfig":
        """Create config optimized for streaming with XLS-R 300M."""
        return cls(
            sample_rate=24000,
            hop_length=480,
            hidden_dim=1024,
            depth=12,
            heads=16,
            rope_dim=64,
            vq_dim=2048,
            semantic_model="facebook/wav2vec2-xls-r-300m",
            semantic_dim=1024,
            streaming_chunk_size=10,
        )


@dataclass
class AudarStreamingState:
    """Complete streaming state for Audar-Codec."""
    # Decoder state
    decoder_state: Optional[DecoderStreamingState] = None
    # Audio buffer for chunking
    audio_buffer: Optional[torch.Tensor] = None
    # Frame counter
    frame_count: int = 0
    
    def reset(self):
        """Reset all state."""
        if self.decoder_state is not None:
            self.decoder_state.reset()
        self.audio_buffer = None
        self.frame_count = 0


class AudarCodec(nn.Module):
    """
    Audar-Codec: Streaming Neural Audio Codec
    
    A streaming-capable neural audio codec with:
    - Causal attention for real-time processing
    - KV-cache for efficient streaming
    - Hierarchical quantization for LLM compatibility
    - Multilingual support via XLS-R 300M
    
    Phases:
    - Phase 1: Streaming decoder (current)
    - Phase 2: Streaming encoder (future)
    - Phase 3: Hierarchical quantization (future)
    
    Example usage:
        ```python
        # Initialize from V1 checkpoint
        codec = AudarCodec.from_v1_checkpoint("neucodec_v1.bin")
        
        # Full sequence decoding
        audio = codec.decode(codes)
        
        # Streaming decoding
        state = codec.get_streaming_state()
        for chunk in code_chunks:
            audio_chunk, state = codec.decode_streaming(chunk, state)
        ```
    """
    
    def __init__(self, config: Optional[AudarCodecConfig] = None):
        super().__init__()
        self.config = config or AudarCodecConfig()
        
        # Decoder (streaming-capable)
        self.decoder = StreamingCodecDecoder(
            hidden_dim=self.config.hidden_dim,
            depth=self.config.depth,
            heads=self.config.heads,
            rope_dim=self.config.rope_dim,
            hop_length=self.config.hop_length,
            vq_dim=self.config.vq_dim,
        )
        
        # Encoder will be added in Phase 2
        self.encoder = None
        
        # Semantic encoder will be added in Phase 2
        self.semantic_encoder = None
    
    @property
    def device(self) -> torch.device:
        """Get model device."""
        return next(self.parameters()).device
    
    @property
    def sample_rate(self) -> int:
        """Audio sample rate."""
        return self.config.sample_rate
    
    @property
    def hop_length(self) -> int:
        """STFT hop length."""
        return self.config.hop_length
    
    @property
    def frame_rate(self) -> float:
        """Codec frame rate in Hz."""
        return self.config.sample_rate / self.config.hop_length
    
    def decode(
        self,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode FSQ codes to audio (full sequence).
        
        Args:
            codes: FSQ codes [B, 1, T] or [B, T, 1]
            
        Returns:
            audio: Reconstructed audio [B, 1, samples]
        """
        audio, _ = self.decoder(codes, from_codes=True)
        return audio
    
    def decode_streaming(
        self,
        codes: torch.Tensor,
        state: Optional[AudarStreamingState] = None,
    ) -> Tuple[torch.Tensor, AudarStreamingState]:
        """
        Decode FSQ codes to audio (streaming).
        
        Args:
            codes: FSQ codes chunk [B, 1, T] or [B, T, 1]
            state: Streaming state from previous call
            
        Returns:
            audio: Audio chunk [B, 1, samples]
            state: Updated streaming state
        """
        if state is None:
            state = AudarStreamingState()
        
        if state.decoder_state is None:
            state.decoder_state = self.decoder.get_initial_state()
        
        audio, state.decoder_state = self.decoder.forward_streaming(
            codes, state.decoder_state, from_codes=True
        )
        
        state.frame_count += codes.shape[-1] if codes.shape[1] == 1 else codes.shape[1]
        
        return audio, state
    
    def decode_embeddings(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decode VQ embeddings to audio (skip quantizer lookup).
        
        Args:
            embeddings: VQ embeddings [B, T, vq_dim]
            
        Returns:
            audio: Reconstructed audio [B, 1, samples]
        """
        audio, _ = self.decoder(embeddings, from_codes=False)
        return audio
    
    def decode_embeddings_streaming(
        self,
        embeddings: torch.Tensor,
        state: Optional[AudarStreamingState] = None,
    ) -> Tuple[torch.Tensor, AudarStreamingState]:
        """
        Decode VQ embeddings to audio (streaming).
        
        Args:
            embeddings: VQ embeddings chunk [B, T, vq_dim]
            state: Streaming state
            
        Returns:
            audio: Audio chunk
            state: Updated state
        """
        if state is None:
            state = AudarStreamingState()
        
        if state.decoder_state is None:
            state.decoder_state = self.decoder.get_initial_state()
        
        audio, state.decoder_state = self.decoder.forward_streaming(
            embeddings, state.decoder_state, from_codes=False
        )
        
        state.frame_count += embeddings.shape[1]
        
        return audio, state
    
    def get_streaming_state(self) -> AudarStreamingState:
        """Create initial streaming state."""
        return AudarStreamingState(
            decoder_state=self.decoder.get_initial_state()
        )
    
    def reset_streaming_state(self, state: AudarStreamingState) -> AudarStreamingState:
        """Reset streaming state for new sequence."""
        state.reset()
        state.decoder_state = self.decoder.get_initial_state()
        return state
    
    @classmethod
    def from_v1_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        config: Optional[AudarCodecConfig] = None,
        device: str = "cpu",
    ) -> "AudarCodec":
        """
        Load Audar-Codec from NeuCodec V1 checkpoint.
        
        Args:
            checkpoint_path: Path to V1 checkpoint
            config: Optional config override
            device: Device to load model to
            
        Returns:
            Initialized AudarCodec with V1 weights
        """
        if config is None:
            config = AudarCodecConfig.from_v1()
        
        # Create model
        model = cls(config)
        
        # Load V1 weights
        v1_state = load_v1_checkpoint(checkpoint_path, device)
        
        # Migrate weights
        migration_config = AudarMigrationConfig(
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            heads=config.heads,
            rope_dim=config.rope_dim,
            hop_length=config.hop_length,
            vq_dim=config.vq_dim,
        )
        audar_weights = migrate_v1_to_audar(v1_state, migration_config)
        
        # Load decoder weights
        decoder_weights = {
            k.replace("decoder.", "") if k.startswith("decoder.") else k: v
            for k, v in audar_weights.items()
        }
        
        missing, unexpected = model.decoder.load_state_dict(decoder_weights, strict=False)
        
        if missing:
            import logging
            logging.warning(f"Missing keys: {len(missing)}")
        
        model.to(device)
        return model
    
    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "neuphonic/audar-codec",
        config: Optional[AudarCodecConfig] = None,
        device: str = "cpu",
        **kwargs,
    ) -> "AudarCodec":
        """
        Load Audar-Codec from HuggingFace Hub.
        
        Args:
            model_id: HuggingFace model ID
            config: Optional config override
            device: Device to load model to
            
        Returns:
            Initialized AudarCodec
        """
        from huggingface_hub import hf_hub_download
        
        # Download checkpoint
        ckpt_path = hf_hub_download(
            repo_id=model_id,
            filename="pytorch_model.bin",
            **kwargs,
        )
        
        return cls.from_v1_checkpoint(ckpt_path, config, device)
    
    def save_pretrained(self, save_path: Union[str, Path]):
        """
        Save model to directory.
        
        Args:
            save_path: Directory to save model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save weights
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")
        
        # Save config
        import json
        config_dict = {
            "sample_rate": self.config.sample_rate,
            "hop_length": self.config.hop_length,
            "hidden_dim": self.config.hidden_dim,
            "depth": self.config.depth,
            "heads": self.config.heads,
            "rope_dim": self.config.rope_dim,
            "vq_dim": self.config.vq_dim,
            "vq_levels": self.config.vq_levels,
            "semantic_model": self.config.semantic_model,
        }
        with open(save_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)
    
    def streaming_benchmark(
        self,
        num_frames: int = 100,
        chunk_size: int = 10,
        device: str = "cuda",
    ) -> Dict[str, float]:
        """
        Benchmark streaming decode performance.
        
        Args:
            num_frames: Total frames to decode
            chunk_size: Frames per chunk
            device: Device to benchmark on
            
        Returns:
            Dictionary with timing statistics
        """
        import time
        
        self.eval()
        self.to(device)
        
        # Generate random codes
        codes = torch.randint(0, 4, (1, 1, num_frames), device=device)
        
        # Warm up
        state = self.get_streaming_state()
        for i in range(0, min(20, num_frames), chunk_size):
            chunk = codes[:, :, i:i+chunk_size]
            _, state = self.decode_streaming(chunk, state)
        
        # Benchmark
        state = self.get_streaming_state()
        torch.cuda.synchronize() if device == "cuda" else None
        
        start = time.perf_counter()
        for i in range(0, num_frames, chunk_size):
            chunk = codes[:, :, i:i+chunk_size]
            _, state = self.decode_streaming(chunk, state)
        
        torch.cuda.synchronize() if device == "cuda" else None
        end = time.perf_counter()
        
        total_time = end - start
        audio_duration = num_frames * self.hop_length / self.sample_rate
        rtf = total_time / audio_duration  # Real-time factor
        
        return {
            "total_time_s": total_time,
            "audio_duration_s": audio_duration,
            "rtf": rtf,
            "frames_per_second": num_frames / total_time,
            "is_realtime": rtf < 1.0,
        }
