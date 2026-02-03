"""
Audar-Codec: Streaming Neural Audio Codec

A streaming-capable, hierarchical neural audio codec built on NeuCodec,
with multilingual support via XLS-R 300M and LLM-compatible tokenization.

Key Features:
- Streaming encoding/decoding with KV-cache
- Hierarchical quantization (12.5Hz coarse + 50Hz fine)
- 128-language multilingual support
- Backward compatible with NeuCodec V1
"""

from .model import AudarCodec, AudarCodecConfig

__all__ = ["AudarCodec", "AudarCodecConfig"]
__version__ = "0.1.0"
