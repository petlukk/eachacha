#!/usr/bin/env python3
"""eachacha_grep — search encrypted files without decrypting to disk.

Usage:
    python3 eachacha_grep.py NEEDLE ENCRYPTED_FILE --key KEY_HEX --nonce NONCE_HEX
        [--counter N] [--max-matches N] [--context]
"""

import argparse
import ctypes as ct
import os
import struct
import sys
from pathlib import Path

_HERE = Path(__file__).parent

# Load shared libraries
try:
    _lib_encrypt = ct.CDLL(str(_HERE / "chacha20.so"))
    _lib_search = ct.CDLL(str(_HERE / "chacha20_search.so"))
except OSError as e:
    print(f"error: could not load shared libraries: {e}", file=sys.stderr)
    print("Make sure chacha20.so and chacha20_search.so are built.", file=sys.stderr)
    sys.exit(1)

# chacha20_encrypt argtypes (10 params)
_lib_encrypt.chacha20_encrypt.argtypes = [
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.c_int32,
    ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_uint8), ct.c_int32,
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_uint8),
    ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),
]
_lib_encrypt.chacha20_encrypt.restype = None

# chacha20_search argtypes (16 params)
_lib_search.chacha20_search.argtypes = [
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
_lib_search.chacha20_search.restype = None


def hex_to_i32_array(hex_str):
    """Convert hex string to little-endian u32 words, then to ctypes i32 array."""
    raw = bytes.fromhex(hex_str)
    words = struct.unpack(f"<{len(raw) // 4}I", raw)
    arr = (ct.c_int32 * len(words))()
    for i, w in enumerate(words):
        arr[i] = ct.c_int32(w & 0xFFFFFFFF).value
    return arr


def do_search(key_arr, nonce_arr, counter, ciphertext_bytes, needle_bytes, max_matches):
    """Run chacha20_search on ciphertext_bytes; return list of match offsets."""
    n = len(ciphertext_bytes)
    needle_len = len(needle_bytes)

    if n == 0:
        return []

    ct_u8 = (ct.c_uint8 * n)(*ciphertext_bytes)
    needle_buf = (ct.c_uint8 * max(needle_len, 1))(*needle_bytes)

    ks_scratch = (ct.c_uint8 * 256)()
    ks_i32 = ct.cast(ks_scratch, ct.POINTER(ct.c_int32))
    ks_u8 = ct.cast(ks_scratch, ct.POINTER(ct.c_uint8))

    ct_i32 = ct.cast(ct_u8, ct.POINTER(ct.c_int32))

    pt_buf = (ct.c_uint8 * 256)()
    pt_i32 = ct.cast(pt_buf, ct.POINTER(ct.c_int32))

    overlap = (ct.c_uint8 * 64)()

    matches = (ct.c_int32 * max_matches)()
    match_count = (ct.c_int32 * 1)()

    _lib_search.chacha20_search(
        key_arr, nonce_arr, ct.c_int32(counter),
        ct.cast(ct_u8, ct.POINTER(ct.c_uint8)), ct.c_int32(n),
        ct.cast(needle_buf, ct.POINTER(ct.c_uint8)), ct.c_int32(needle_len),
        ks_i32, ks_u8,
        ct_i32,
        ct.cast(pt_buf, ct.POINTER(ct.c_uint8)), pt_i32,
        ct.cast(overlap, ct.POINTER(ct.c_uint8)),
        matches, ct.c_int32(max_matches),
        match_count,
    )

    count = match_count[0]
    return [matches[i] for i in range(count)]


def decrypt_context(key_arr, nonce_arr, base_counter, ciphertext_bytes, ctx_start, ctx_end):
    """Decrypt a slice of ciphertext using block-aligned ChaCha20 (XOR is its own inverse).

    Returns the decrypted bytes for the range [ctx_start, ctx_end).
    """
    n = len(ciphertext_bytes)
    ctx_start = max(0, ctx_start)
    ctx_end = min(n, ctx_end)
    if ctx_start >= ctx_end:
        return b""

    # Align start to 64-byte block boundary
    block_counter = base_counter + (ctx_start // 64)
    block_offset = ctx_start % 64
    aligned_start = ctx_start - block_offset

    slice_len = ctx_end - aligned_start

    # Build input buffer from aligned_start
    ct_slice = (ct.c_uint8 * slice_len)(*ciphertext_bytes[aligned_start:ctx_end])
    pt_out = (ct.c_uint8 * slice_len)()

    ks_scratch = (ct.c_uint8 * 64)()
    ks_i32 = ct.cast(ks_scratch, ct.POINTER(ct.c_int32))
    ks_u8 = ct.cast(ks_scratch, ct.POINTER(ct.c_uint8))
    pt_i32 = ct.cast(ct_slice, ct.POINTER(ct.c_int32))
    ct_i32 = ct.cast(pt_out, ct.POINTER(ct.c_int32))

    _lib_encrypt.chacha20_encrypt(
        key_arr, nonce_arr, ct.c_int32(block_counter),
        ct_slice, pt_out, ct.c_int32(slice_len),
        ks_i32, ks_u8,
        pt_i32, ct_i32,
    )

    # Extract the portion we actually want (skip the block_offset prefix)
    result = bytes(pt_out[block_offset:])
    return result


def format_context(data_bytes):
    """Format context bytes for display, replacing non-printable chars with '.'."""
    result = []
    for b in data_bytes:
        if 32 <= b < 127:
            result.append(chr(b))
        else:
            result.append(".")
    return "".join(result)


def main():
    parser = argparse.ArgumentParser(
        description="Search for a plaintext needle inside a ChaCha20-encrypted file."
    )
    parser.add_argument("needle", help="Plaintext string to search for")
    parser.add_argument("encrypted_file", help="Path to the encrypted file")
    parser.add_argument("--key", required=True, metavar="KEY_HEX",
                        help="32-byte key as 64 hex chars")
    parser.add_argument("--nonce", required=True, metavar="NONCE_HEX",
                        help="12-byte nonce as 24 hex chars")
    parser.add_argument("--counter", type=int, default=1, metavar="N",
                        help="Initial block counter (default: 1)")
    parser.add_argument("--max-matches", type=int, default=1000, metavar="N",
                        help="Maximum number of matches to report (default: 1000)")
    parser.add_argument("--context", action="store_true",
                        help="Show ±40 bytes of decrypted context around each match "
                             "(only when ≤100 matches)")
    args = parser.parse_args()

    # Validate and parse key
    key_hex = args.key.strip()
    if len(key_hex) != 64:
        print(f"error: --key must be 64 hex characters (32 bytes), got {len(key_hex)}",
              file=sys.stderr)
        sys.exit(1)
    try:
        key_arr = hex_to_i32_array(key_hex)
    except ValueError as e:
        print(f"error: invalid --key hex: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate and parse nonce
    nonce_hex = args.nonce.strip()
    if len(nonce_hex) != 24:
        print(f"error: --nonce must be 24 hex characters (12 bytes), got {len(nonce_hex)}",
              file=sys.stderr)
        sys.exit(1)
    try:
        nonce_arr = hex_to_i32_array(nonce_hex)
    except ValueError as e:
        print(f"error: invalid --nonce hex: {e}", file=sys.stderr)
        sys.exit(1)

    # Read the encrypted file
    enc_path = args.encrypted_file
    if not os.path.isfile(enc_path):
        print(f"error: file not found: {enc_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(enc_path, "rb") as f:
            ciphertext = f.read()
    except OSError as e:
        print(f"error: could not read file: {e}", file=sys.stderr)
        sys.exit(1)

    needle_bytes = args.needle.encode("utf-8")
    needle_len = len(needle_bytes)

    if needle_len == 0:
        print("matches: 0")
        print("offsets: []")
        return

    file_size = len(ciphertext)
    print(f"file:    {enc_path} ({file_size} bytes)")
    print(f"needle:  {args.needle!r} ({needle_len} bytes)")
    print(f"counter: {args.counter}")

    # Run the search
    offsets = do_search(
        key_arr, nonce_arr, args.counter,
        ciphertext, needle_bytes, args.max_matches
    )

    count = len(offsets)
    print(f"matches: {count}")

    if count == 0:
        print("offsets: []")
        return

    # Print offsets (truncated if large)
    if count <= 20:
        print(f"offsets: {offsets}")
    else:
        print(f"offsets: {offsets[:20]} ... (+{count - 20} more)")

    # Context display
    if args.context and count <= 100:
        context_radius = 40
        print()
        for i, offset in enumerate(offsets):
            ctx_start = offset - context_radius
            ctx_end = offset + needle_len + context_radius
            ctx_bytes = decrypt_context(
                key_arr, nonce_arr, args.counter,
                ciphertext, ctx_start, ctx_end
            )
            # Figure out where needle starts within the returned context bytes
            actual_start = max(0, offset - context_radius)
            needle_pos_in_ctx = offset - actual_start
            display = format_context(ctx_bytes)
            # Highlight needle position with markers
            pre = display[:needle_pos_in_ctx]
            match_str = display[needle_pos_in_ctx:needle_pos_in_ctx + needle_len]
            post = display[needle_pos_in_ctx + needle_len:]
            print(f"  [{i+1}] offset={offset}: ...{pre}>>>{match_str}<<<{post}...")
    elif args.context and count > 100:
        print()
        print(f"  (skipping context display: {count} matches exceeds 100-match limit)")


if __name__ == "__main__":
    main()
