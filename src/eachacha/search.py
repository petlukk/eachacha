"""Searchable cipher — search encrypted data without decrypting to disk."""

import ctypes as ct
from dataclasses import dataclass, field
from typing import List, Optional, Union

from ._lib import get_search_lib, get_search_v2_lib, key_nonce_from_bytes


@dataclass
class SearchResult:
    """Result from searching encrypted data."""
    match_count: int
    offsets: List[int]
    needle_ids: List[int] = field(default_factory=list)
    lines: List[bytes] = field(default_factory=list)


def search(ciphertext: bytes, needles: Union[bytes, List[bytes]], key: bytes, nonce: bytes,
           counter: int = 1, max_matches: int = 100000,
           max_line_len: int = 1024, window_size: int = 4096) -> SearchResult:
    """Search encrypted data without decrypting to disk.

    Accepts a single needle (bytes) or multiple needles (list of bytes).
    Automatically selects the fastest kernel: v1 for single needle,
    v2 for multiple needles (with context-line extraction).

    Args:
        ciphertext: Encrypted data
        needles: Pattern (bytes) or list of patterns (list of bytes)
        key: 32-byte ChaCha20 key
        nonce: 12-byte nonce
        counter: Initial block counter
        max_matches: Maximum matches to return
        max_line_len: Max line length for context extraction (v2 only)
        window_size: Decrypt window size (v2 only, default 4096)

    Returns:
        SearchResult with match_count, offsets, needle_ids, and lines
    """
    # Normalize input
    if isinstance(needles, bytes):
        needle_list = [needles]
    else:
        needle_list = list(needles)

    size = len(ciphertext)
    if size == 0 or not needle_list:
        return SearchResult(match_count=0, offsets=[])

    # Single needle → v1 kernel (faster, no line extraction overhead)
    if len(needle_list) == 1:
        return _search_v1(ciphertext, needle_list[0], key, nonce, counter, max_matches)

    # Multiple needles → v2 kernel (one decrypt pass + context lines)
    return _search_v2(ciphertext, needle_list, key, nonce, counter,
                      max_matches, max_line_len, window_size)


def _search_v1(ciphertext, needle, key, nonce, counter, max_matches):
    """Single-needle search via v1 kernel."""
    size = len(ciphertext)
    if len(needle) == 0 or len(needle) > 64:
        return SearchResult(match_count=0, offsets=[])

    lib = get_search_lib()
    key_arr, nonce_arr = key_nonce_from_bytes(key, nonce)
    ct_buf = (ct.c_uint8 * size)(*ciphertext)
    needle_buf = (ct.c_uint8 * len(needle))(*needle)
    ks = (ct.c_uint8 * 256)()
    ks_i32 = ct.cast(ks, ct.POINTER(ct.c_int32))
    ks_u8 = ct.cast(ks, ct.POINTER(ct.c_uint8))
    ct_i32 = ct.cast(ct_buf, ct.POINTER(ct.c_int32))
    pt_buf = (ct.c_uint8 * 256)()
    pt_i32 = ct.cast(pt_buf, ct.POINTER(ct.c_int32))
    overlap = (ct.c_uint8 * 64)()
    matches = (ct.c_int32 * max_matches)()
    match_count = (ct.c_int32 * 1)()

    lib.chacha20_search(
        key_arr, nonce_arr, ct.c_int32(counter),
        ct_buf, ct.c_int32(size),
        needle_buf, ct.c_int32(len(needle)),
        ks_i32, ks_u8, ct_i32,
        pt_buf, pt_i32, overlap,
        matches, ct.c_int32(max_matches), match_count)

    mc = match_count[0]
    return SearchResult(
        match_count=mc,
        offsets=[matches[i] for i in range(mc)],
        needle_ids=[0] * mc)


def _search_v2(ciphertext, needle_list, key, nonce, counter,
               max_matches, max_line_len, window_size):
    """Multi-needle search via v2 kernel with context-line extraction."""
    size = len(ciphertext)

    lib = get_search_v2_lib()
    key_arr, nonce_arr = key_nonce_from_bytes(key, nonce)

    # Pack needles
    packed = b""
    offsets_list = []
    lens_list = []
    for n in needle_list:
        offsets_list.append(len(packed))
        lens_list.append(len(n))
        packed += n

    ct_buf = (ct.c_uint8 * size)(*ciphertext)
    needles_buf = (ct.c_uint8 * len(packed))(*packed)
    offsets_arr = (ct.c_int32 * len(offsets_list))(*offsets_list)
    lens_arr = (ct.c_int32 * len(lens_list))(*lens_list)
    ks = (ct.c_uint8 * 256)()
    ks_i32 = ct.cast(ks, ct.POINTER(ct.c_int32))
    ks_u8 = ct.cast(ks, ct.POINTER(ct.c_uint8))
    ct_i32 = ct.cast(ct_buf, ct.POINTER(ct.c_int32))
    pt_buf = (ct.c_uint8 * max(window_size, 256))()
    pt_i32 = ct.cast(pt_buf, ct.POINTER(ct.c_int32))
    overlap = (ct.c_uint8 * 64)()
    lines_buf_cap = max_matches * max_line_len
    lines_buf = (ct.c_uint8 * max(lines_buf_cap, 1))()
    line_offsets = (ct.c_int32 * max_matches)()
    line_lens = (ct.c_int32 * max_matches)()
    match_offsets = (ct.c_int32 * max_matches)()
    needle_ids = (ct.c_int32 * max_matches)()
    match_count = (ct.c_int32 * 1)()
    lines_written = (ct.c_int32 * 1)()

    lib.chacha20_search_v2(
        key_arr, nonce_arr, ct.c_int32(counter),
        ct_buf, ct.c_int32(size),
        ks_i32, ks_u8, ct_i32,
        pt_buf, pt_i32, overlap,
        needles_buf, offsets_arr, lens_arr, ct.c_int32(len(needle_list)),
        lines_buf, ct.c_int32(lines_buf_cap),
        line_offsets, line_lens, match_offsets, needle_ids,
        ct.c_int32(max_matches), ct.c_int32(max_line_len), ct.c_int32(window_size),
        match_count, lines_written)

    mc = match_count[0]
    lw = lines_written[0]
    lines_out = []
    for i in range(lw):
        lo = line_offsets[i]
        ll = line_lens[i]
        lines_out.append(bytes(lines_buf[lo:lo + ll]))

    return SearchResult(
        match_count=mc,
        offsets=[match_offsets[i] for i in range(mc)],
        needle_ids=[needle_ids[i] for i in range(mc)],
        lines=lines_out)
