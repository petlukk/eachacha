"""ChaCha20 encrypt/decrypt via Eä SIMD kernel."""

import ctypes as ct

from ._lib import get_encrypt_lib, key_nonce_from_bytes


def encrypt(plaintext: bytes, key: bytes, nonce: bytes, counter: int = 1) -> bytes:
    """Encrypt plaintext with ChaCha20 (RFC 7539).

    Args:
        plaintext: Data to encrypt
        key: 32-byte key
        nonce: 12-byte nonce
        counter: Initial block counter (default 1)

    Returns:
        Ciphertext bytes (same length as plaintext)
    """
    size = len(plaintext)
    if size == 0:
        return b""

    lib = get_encrypt_lib()
    key_arr, nonce_arr = key_nonce_from_bytes(key, nonce)
    pt = (ct.c_uint8 * size)(*plaintext)
    ct_buf = (ct.c_uint8 * size)()
    scratch = (ct.c_uint8 * 256)()
    ks_i32 = ct.cast(scratch, ct.POINTER(ct.c_int32))
    ks_u8 = ct.cast(scratch, ct.POINTER(ct.c_uint8))
    pt_i32 = ct.cast(pt, ct.POINTER(ct.c_int32))
    ct_i32 = ct.cast(ct_buf, ct.POINTER(ct.c_int32))

    lib.chacha20_encrypt(key_arr, nonce_arr, ct.c_int32(counter),
                         pt, ct_buf, ct.c_int32(size),
                         ks_i32, ks_u8, pt_i32, ct_i32)
    return bytes(ct_buf)


def decrypt(ciphertext: bytes, key: bytes, nonce: bytes, counter: int = 1) -> bytes:
    """Decrypt ciphertext with ChaCha20 (XOR is its own inverse)."""
    return encrypt(ciphertext, key, nonce, counter)
