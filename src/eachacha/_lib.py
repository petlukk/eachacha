"""Load pre-built shared libraries from the lib/ directory."""

import ctypes as ct
import sys
from pathlib import Path

_LIB_DIR = Path(__file__).parent / "lib"
_EXT = ".dll" if sys.platform == "win32" else ".so"


def _load(name):
    path = _LIB_DIR / f"{name}{_EXT}"
    if not path.exists():
        raise RuntimeError(
            f"Shared library not found: {path}. "
            f"Rebuild with: EA=ea ./build.sh"
        )
    return ct.CDLL(str(path))


# Lazy loading — only load when first accessed
_cache = {}


def get_encrypt_lib():
    if "encrypt" not in _cache:
        lib = _load("chacha20")
        lib.chacha20_encrypt.argtypes = [
            ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.c_int32,
            ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_uint8), ct.c_int32,
            ct.POINTER(ct.c_int32), ct.POINTER(ct.c_uint8),
            ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),
        ]
        lib.chacha20_encrypt.restype = None
        _cache["encrypt"] = lib
    return _cache["encrypt"]


def get_search_lib():
    if "search" not in _cache:
        lib = _load("chacha20_search")
        lib.chacha20_search.argtypes = [
            ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.c_int32,
            ct.POINTER(ct.c_uint8), ct.c_int32,
            ct.POINTER(ct.c_uint8), ct.c_int32,
            ct.POINTER(ct.c_int32), ct.POINTER(ct.c_uint8),
            ct.POINTER(ct.c_int32),
            ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_int32),
            ct.POINTER(ct.c_uint8),
            ct.POINTER(ct.c_int32), ct.c_int32,
            ct.POINTER(ct.c_int32),
        ]
        lib.chacha20_search.restype = None
        _cache["search"] = lib
    return _cache["search"]


def get_search_v2_lib():
    if "search_v2" not in _cache:
        lib = _load("chacha20_search_v2")
        lib.chacha20_search_v2.argtypes = [
            ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.c_int32,
            ct.POINTER(ct.c_uint8), ct.c_int32,
            ct.POINTER(ct.c_int32), ct.POINTER(ct.c_uint8),
            ct.POINTER(ct.c_int32),
            ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_int32),
            ct.POINTER(ct.c_uint8),
            ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_int32),
            ct.POINTER(ct.c_int32), ct.c_int32,
            ct.POINTER(ct.c_uint8), ct.c_int32,
            ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),
            ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),
            ct.c_int32, ct.c_int32, ct.c_int32,
            ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),
        ]
        lib.chacha20_search_v2.restype = None
        _cache["search_v2"] = lib
    return _cache["search_v2"]


def to_i32_array(values):
    arr = (ct.c_int32 * len(values))()
    for i, v in enumerate(values):
        arr[i] = ct.c_int32(v & 0xFFFFFFFF).value
    return arr


def key_nonce_from_bytes(key_bytes, nonce_bytes):
    """Convert 32-byte key and 12-byte nonce to ctypes i32 arrays."""
    import struct
    key_words = struct.unpack("<8I", key_bytes)
    nonce_words = struct.unpack("<3I", nonce_bytes)
    return to_i32_array(key_words), to_i32_array(nonce_words)
