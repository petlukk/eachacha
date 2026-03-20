#!/usr/bin/env python3
"""eachacha_grep — search encrypted files without decrypting to disk.

Usage:
    python3 eachacha_grep.py NEEDLE [NEEDLE ...] ENCRYPTED_FILE --key KEY_HEX --nonce NONCE_HEX
        [--counter N] [--max-matches N] [--context]

With a single needle, the v1 kernel is used.
With 2+ needles, the v2 kernel is used (fused multi-needle + context-line extraction).
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

# Load v2 library lazily (only needed for multi-needle)
_lib_search_v2 = None

def _load_v2():
    global _lib_search_v2
    if _lib_search_v2 is not None:
        return _lib_search_v2
    try:
        lib = ct.CDLL(str(_HERE / "chacha20_search_v2.so"))
    except OSError as e:
        print(f"error: could not load chacha20_search_v2.so: {e}", file=sys.stderr)
        print("Make sure chacha20_search_v2.so is built.", file=sys.stderr)
        sys.exit(1)
    # argtypes from auto-generated bindings (26 params)
    lib.chacha20_search_v2.argtypes = [
        ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32), ct.c_int32,  # key, nonce, ctr
        ct.POINTER(ct.c_uint8), ct.c_int32,                           # ct_u8, len
        ct.POINTER(ct.c_int32), ct.POINTER(ct.c_uint8),               # ks_i32, ks_u8
        ct.POINTER(ct.c_int32),                                        # ct_i32
        ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_int32),               # pt_buf, pt_i32
        ct.POINTER(ct.c_uint8),                                        # overlap
        ct.POINTER(ct.c_uint8), ct.POINTER(ct.c_int32),               # needles, needle_offsets
        ct.POINTER(ct.c_int32), ct.c_int32,                           # needle_lens, needle_count
        ct.POINTER(ct.c_uint8), ct.c_int32,                           # lines_buf, lines_buf_cap
        ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),               # line_offsets, line_lens
        ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),               # match_offsets, needle_ids
        ct.c_int32, ct.c_int32, ct.c_int32,                           # max_matches, max_line_len, window_size
        ct.POINTER(ct.c_int32), ct.POINTER(ct.c_int32),               # match_count, lines_written
    ]
    lib.chacha20_search_v2.restype = None
    _lib_search_v2 = lib
    return lib

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


def do_search_v2(key_arr, nonce_arr, counter, ciphertext_bytes, needle_list, max_matches,
                 max_line_len=1024, window_size=4096):
    """Run chacha20_search_v2 on ciphertext_bytes for multiple needles.

    Returns (match_offsets, needle_ids, lines, lines_written_count).
    """
    lib = _load_v2()
    size = len(ciphertext_bytes)
    if size == 0:
        return [], [], [], 0

    ct_u8 = (ct.c_uint8 * size)(*ciphertext_bytes)

    # Pack needles into a concatenated buffer with offset/length arrays
    packed = b""
    offsets = []
    lens = []
    for nb in needle_list:
        offsets.append(len(packed))
        lens.append(len(nb))
        packed += nb
    if not packed:
        packed = b"\x00"

    needles_buf = (ct.c_uint8 * len(packed))(*packed)
    offsets_arr = (ct.c_int32 * max(len(offsets), 1))(*offsets) if offsets else (ct.c_int32 * 1)()
    lens_arr = (ct.c_int32 * max(len(lens), 1))(*lens) if lens else (ct.c_int32 * 1)()

    # Scratch buffers
    ks = (ct.c_uint8 * 256)()
    ks_i32 = ct.cast(ks, ct.POINTER(ct.c_int32))
    ks_u8 = ct.cast(ks, ct.POINTER(ct.c_uint8))
    ct_i32 = ct.cast(ct_u8, ct.POINTER(ct.c_int32))
    pt_buf = (ct.c_uint8 * max(window_size, 256))()
    pt_i32 = ct.cast(pt_buf, ct.POINTER(ct.c_int32))
    overlap = (ct.c_uint8 * 64)()

    lines_buf_cap = max(max_matches * max_line_len, 1)
    lines_buf = (ct.c_uint8 * lines_buf_cap)()
    line_offsets = (ct.c_int32 * max_matches)()
    line_lens_arr = (ct.c_int32 * max_matches)()
    match_offsets = (ct.c_int32 * max_matches)()
    needle_ids = (ct.c_int32 * max_matches)()
    match_count = (ct.c_int32 * 1)()
    lines_written = (ct.c_int32 * 1)()

    lib.chacha20_search_v2(
        key_arr, nonce_arr, ct.c_int32(counter),
        ct.cast(ct_u8, ct.POINTER(ct.c_uint8)), ct.c_int32(size),
        ks_i32, ks_u8, ct_i32,
        ct.cast(pt_buf, ct.POINTER(ct.c_uint8)), pt_i32, ct.cast(overlap, ct.POINTER(ct.c_uint8)),
        ct.cast(needles_buf, ct.POINTER(ct.c_uint8)), offsets_arr, lens_arr,
        ct.c_int32(len(needle_list)),
        ct.cast(lines_buf, ct.POINTER(ct.c_uint8)), ct.c_int32(lines_buf_cap),
        line_offsets, line_lens_arr,
        match_offsets, needle_ids,
        ct.c_int32(max_matches), ct.c_int32(max_line_len), ct.c_int32(window_size),
        match_count, lines_written,
    )

    mc = match_count[0]
    lw = lines_written[0]
    offsets_out = [match_offsets[i] for i in range(mc)]
    ids_out = [needle_ids[i] for i in range(mc)]
    lines_out = []
    for i in range(lw):
        lo = line_offsets[i]
        ll = line_lens_arr[i]
        lines_out.append(bytes(lines_buf[lo:lo + ll]))
    return offsets_out, ids_out, lines_out, lw


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
        description="Search for plaintext needles inside a ChaCha20-encrypted file. "
                    "With 1 needle uses the v1 kernel; with 2+ needles uses the v2 multi-needle kernel.\n\n"
                    "Usage: eachacha_grep.py NEEDLE [NEEDLE ...] ENCRYPTED_FILE --key HEX --nonce HEX",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("needles", nargs='+',
                        metavar="NEEDLE_OR_FILE",
                        help="One or more plaintext needles followed by the encrypted file path. "
                             "The last positional argument is the encrypted file.")
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
                             "(only when ≤100 matches, v1 path only)")
    args = parser.parse_args()

    # Split positional args: last one is the file, the rest are needles
    if len(args.needles) < 2:
        parser.error("At least one NEEDLE and one ENCRYPTED_FILE must be provided.")
    needle_strs = args.needles[:-1]
    enc_path = args.needles[-1]

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
    if not os.path.isfile(enc_path):
        print(f"error: file not found: {enc_path}", file=sys.stderr)
        sys.exit(1)

    try:
        with open(enc_path, "rb") as f:
            ciphertext = f.read()
    except OSError as e:
        print(f"error: could not read file: {e}", file=sys.stderr)
        sys.exit(1)

    file_size = len(ciphertext)
    print(f"file:    {enc_path} ({file_size} bytes)")
    print(f"counter: {args.counter}")

    # -----------------------------------------------------------------------
    # Single-needle path: use v1 kernel
    # -----------------------------------------------------------------------
    if len(needle_strs) == 1:
        needle_bytes = needle_strs[0].encode("utf-8")
        needle_len = len(needle_bytes)
        print(f"needle:  {needle_strs[0]!r} ({needle_len} bytes)")
        print(f"kernel:  v1")

        if needle_len == 0:
            print("matches: 0")
            print("offsets: []")
            return

        offsets = do_search(
            key_arr, nonce_arr, args.counter,
            ciphertext, needle_bytes, args.max_matches
        )

        count = len(offsets)
        print(f"matches: {count}")

        if count == 0:
            print("offsets: []")
            return

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
                actual_start = max(0, offset - context_radius)
                needle_pos_in_ctx = offset - actual_start
                display = format_context(ctx_bytes)
                pre = display[:needle_pos_in_ctx]
                match_str = display[needle_pos_in_ctx:needle_pos_in_ctx + needle_len]
                post = display[needle_pos_in_ctx + needle_len:]
                print(f"  [{i+1}] offset={offset}: ...{pre}>>>{match_str}<<<{post}...")
        elif args.context and count > 100:
            print()
            print(f"  (skipping context display: {count} matches exceeds 100-match limit)")

    # -----------------------------------------------------------------------
    # Multi-needle path: use v2 kernel
    # -----------------------------------------------------------------------
    else:
        needle_list = [s.encode("utf-8") for s in needle_strs]
        needle_summary = ", ".join(repr(s) for s in needle_strs)
        print(f"needles: [{needle_summary}] ({len(needle_list)} needles)")
        print(f"kernel:  v2")

        match_offsets, needle_ids, lines, lw = do_search_v2(
            key_arr, nonce_arr, args.counter,
            ciphertext, needle_list, args.max_matches
        )

        count = len(match_offsets)
        print(f"matches: {count}")

        if count == 0:
            print("offsets: []")
            return

        # Display results sorted by offset, with [NEEDLE] prefix
        paired = sorted(zip(match_offsets, needle_ids, range(count)), key=lambda x: x[0])
        print()
        for rank, (offset, nid, orig_idx) in enumerate(paired):
            needle_label = needle_strs[nid] if nid < len(needle_strs) else f"#{nid}"
            # Find corresponding line if available (lines are ordered by match, not sorted)
            if orig_idx < lw:
                line_text = lines[orig_idx].decode("utf-8", errors="replace")
                print(f"  [{rank+1}] [{needle_label}] offset={offset}: {line_text}")
            else:
                print(f"  [{rank+1}] [{needle_label}] offset={offset}")


if __name__ == "__main__":
    main()
