"""eachacha: ChaCha20 SIMD kernels in Eä — encrypt, fused stats, and searchable cipher."""

from .search import search
from .encrypt import encrypt, decrypt

__version__ = "1.0.3"

__all__ = ["encrypt", "decrypt", "search"]
