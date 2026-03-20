"""eachacha: ChaCha20 SIMD kernels in Eä — encrypt, fused stats, and searchable cipher."""

from .search import search, search_multi
from .encrypt import encrypt, decrypt

__version__ = "1.0.0"

__all__ = ["encrypt", "decrypt", "search", "search_multi"]
