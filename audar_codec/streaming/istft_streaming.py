"""
Streaming ISTFT with overlap-add buffer.

Enables frame-by-frame audio synthesis without needing the full sequence.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class StreamingISTFT(nn.Module):
    """
    Streaming Inverse Short-Time Fourier Transform.
    
    Processes spectrogram frames incrementally using overlap-add synthesis.
    Maintains a buffer for the overlap region between consecutive frames.
    
    Args:
        n_fft: FFT size
        hop_length: Hop length between frames
        win_length: Window length (defaults to n_fft)
        padding: Padding mode ("same" or "center")
    """
    
    def __init__(
        self,
        n_fft: int,
        hop_length: int,
        win_length: Optional[int] = None,
        padding: str = "same",
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length if win_length is not None else n_fft
        self.padding = padding
        
        # Register window as buffer
        window = torch.hann_window(self.win_length)
        self.register_buffer("window", window)
        
        # Overlap size for streaming
        self.overlap_size = self.win_length - self.hop_length
    
    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        """
        Full ISTFT (for training/non-streaming).
        
        Args:
            spec: Complex spectrogram [batch, n_fft//2 + 1, frames]
            
        Returns:
            audio: Reconstructed waveform [batch, samples]
        """
        if self.padding == "center":
            return torch.istft(
                spec,
                self.n_fft,
                self.hop_length,
                self.win_length,
                self.window,
                center=True,
            )
        elif self.padding == "same":
            return self._istft_same_padding(spec)
        else:
            raise ValueError(f"Padding must be 'center' or 'same', got {self.padding}")
    
    def _istft_same_padding(self, spec: torch.Tensor) -> torch.Tensor:
        """
        ISTFT with 'same' padding (matching NeuCodec's implementation).
        
        This is a custom implementation since torch.istft doesn't support
        'same' padding due to NOLA constraints at edges.
        """
        assert spec.dim() == 3, "Expected 3D tensor [batch, freq, frames]"
        B, N, T = spec.shape
        
        pad = (self.win_length - self.hop_length) // 2
        
        # Inverse FFT
        ifft = torch.fft.irfft(spec, self.n_fft, dim=1, norm="backward")
        ifft = ifft * self.window[None, :, None]
        
        # Overlap and add using fold
        output_size = (T - 1) * self.hop_length + self.win_length
        y = F.fold(
            ifft,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        )[:, 0, 0, pad:-pad]
        
        # Window envelope normalization
        window_sq = self.window.square().expand(1, T, -1).transpose(1, 2)
        window_envelope = F.fold(
            window_sq,
            output_size=(1, output_size),
            kernel_size=(1, self.win_length),
            stride=(1, self.hop_length),
        ).squeeze()[pad:-pad]
        
        # Normalize
        y = y / (window_envelope + 1e-11)
        
        return y
    
    def forward_streaming(
        self,
        spec_chunk: torch.Tensor,
        buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Streaming ISTFT with overlap-add buffer.
        
        Processes one or more spectrogram frames and returns audio samples.
        
        Args:
            spec_chunk: Spectrogram frames [batch, n_fft//2 + 1, num_frames]
            buffer: Overlap buffer from previous call [batch, overlap_size]
                   or None to initialize
            
        Returns:
            audio_chunk: Output audio samples [batch, num_frames * hop_length]
            new_buffer: Updated overlap buffer [batch, overlap_size]
        """
        B, N, T_frames = spec_chunk.shape
        
        # Initialize buffer if needed
        if buffer is None:
            buffer = torch.zeros(
                B, self.overlap_size,
                device=spec_chunk.device,
                dtype=torch.float32,
            )
        
        # Inverse FFT each frame
        ifft_frames = torch.fft.irfft(spec_chunk, self.n_fft, dim=1, norm="backward")
        # ifft_frames: [B, n_fft, T_frames]
        
        # Truncate to window length and apply window
        ifft_frames = ifft_frames[:, :self.win_length, :] * self.window[None, :, None]
        # ifft_frames: [B, win_length, T_frames]
        
        # Overlap-add frame by frame
        output_samples = []
        
        for t in range(T_frames):
            frame = ifft_frames[:, :, t]  # [B, win_length]
            
            # Add overlap from buffer to beginning of frame
            frame[:, :self.overlap_size] = frame[:, :self.overlap_size] + buffer
            
            # Output the non-overlapping part (first hop_length samples)
            output_samples.append(frame[:, :self.hop_length])
            
            # Update buffer with the overlap part (last overlap_size samples)
            buffer = frame[:, self.hop_length:].clone()
        
        # Concatenate output samples
        audio_chunk = torch.cat(output_samples, dim=1)  # [B, T_frames * hop_length]
        
        return audio_chunk, buffer
    
    def get_initial_buffer(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """
        Create initial zero buffer for streaming.
        
        Args:
            batch_size: Batch size
            device: Device for buffer
            dtype: Data type for buffer
            
        Returns:
            Zero buffer [batch_size, overlap_size]
        """
        return torch.zeros(
            batch_size, self.overlap_size,
            device=device, dtype=dtype,
        )


class StreamingISTFTHead(nn.Module):
    """
    ISTFT head with learnable projection for spectrogram prediction.
    
    Takes latent features and predicts magnitude and phase for ISTFT.
    Supports both full-sequence and streaming inference.
    
    Args:
        dim: Input feature dimension
        n_fft: FFT size
        hop_length: Hop length between frames
        padding: Padding mode ("same" or "center")
    """
    
    def __init__(
        self,
        dim: int,
        n_fft: int,
        hop_length: int,
        padding: str = "same",
    ):
        super().__init__()
        self.dim = dim
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        # Output projection: predict magnitude and phase
        out_dim = n_fft + 2  # n_fft//2 + 1 for mag, same for phase
        self.out = nn.Linear(dim, out_dim)
        
        # Streaming ISTFT
        self.istft = StreamingISTFT(
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            padding=padding,
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass (for training/non-streaming).
        
        Args:
            x: Input features [batch, seq_len, dim]
            
        Returns:
            audio: Reconstructed waveform [batch, 1, samples]
            x_pred: Predicted spectrogram features (for loss computation)
        """
        # Project to mag/phase
        x_pred = self.out(x)  # [B, T, n_fft + 2]
        x_pred = x_pred.transpose(1, 2)  # [B, n_fft + 2, T]
        
        # Split into magnitude and phase
        mag, p = x_pred.chunk(2, dim=1)
        
        # Exponential magnitude with clipping for stability
        mag = torch.exp(mag).clamp(max=1e2)
        
        # Convert to complex spectrogram
        x_real = mag * torch.cos(p)
        x_imag = mag * torch.sin(p)
        S = torch.complex(x_real, x_imag)
        
        # ISTFT
        audio = self.istft(S)
        
        return audio.unsqueeze(1), x_pred
    
    def forward_streaming(
        self,
        x: torch.Tensor,
        istft_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Streaming forward pass.
        
        Args:
            x: Input features chunk [batch, chunk_len, dim]
            istft_buffer: Overlap buffer from previous call
            
        Returns:
            audio: Output audio chunk [batch, 1, chunk_len * hop_length]
            new_buffer: Updated ISTFT buffer
        """
        # Project to mag/phase
        x_pred = self.out(x)  # [B, T_chunk, n_fft + 2]
        x_pred = x_pred.transpose(1, 2)  # [B, n_fft + 2, T_chunk]
        
        # Split into magnitude and phase
        mag, p = x_pred.chunk(2, dim=1)
        
        # Exponential magnitude with clipping
        mag = torch.exp(mag).clamp(max=1e2)
        
        # Convert to complex spectrogram
        x_real = mag * torch.cos(p)
        x_imag = mag * torch.sin(p)
        S = torch.complex(x_real, x_imag)
        
        # Streaming ISTFT
        audio, new_buffer = self.istft.forward_streaming(S, istft_buffer)
        
        return audio.unsqueeze(1), new_buffer
