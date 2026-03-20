# Eä Kernel Optimization — Searchable Cipher v2

You are optimizing a fused ChaCha20-Decrypt + Multi-Needle-Search + Context-Line-Extraction kernel.
The kernel decrypts encrypted data into a window, searches for multiple string patterns using SIMD,
and extracts matched log lines — all in a single streaming pass.

Your goal: produce the fastest correct kernel in the fewest lines of code.

## Your Task

Edit the kernel to improve throughput (GB/s). You MUST output a HYPOTHESIS line and then the complete kernel.ea in a code fence.

## Rules

1. Only valid Eä syntax. Do not invent intrinsics or syntax that doesn't exist.
2. Correctness is non-negotiable. Match offsets must exactly match the Python reference.
3. One change per iteration. State your hypothesis clearly.
4. The `chacha20_search_v2` export function signature must not change (26 parameters).
5. Internal helper function signatures may change, but exports must stay the same.
6. No dead code. No comments longer than one line.

## Architecture

The kernel has three phases per window:
1. **Decrypt** — 3-tier ChaCha20: 4-block ILP (256B) → single-block (64B) → sub-block tail
2. **Search** — multi-needle SIMD: `.==` + `movemask` per unique first-byte, OR:ed bitmasks, scalar verify
3. **Extract** — find \n boundaries backward/forward via `.==` + `movemask`, copy line to output buffer

The decrypt phase is the most expensive (~70% of time). The search phase is fast (most chunks skipped by SIMD). Line extraction is rare (only on matches).

## Key Constraint

`pt_buf` and `pt_i32` are aliasing pointers to the same buffer. They must NOT use `*restrict`.
All other pointer parameters can use `*restrict` or `*restrict mut` for alias optimization.

## Strategy Space

**Decrypt optimizations (highest impact):**
- Window size (current: 4096). Larger windows = fewer transitions but more zeroing
- Skip zeroing (security trade-off for benchmarks)
- 4-block ILP is already optimal. Focus on reducing overhead between decrypt iterations.
- `prefetch(ct_u8, offset + 256)` to prefetch next ciphertext block
- `stream_store` for zeroing (avoids cache pollution)

**Search optimizations:**
- Current: `bits = bits + movemask(...)` for OR:ed bitmasks. Use `bits = bits | movemask(...)` instead (correct OR, may help branch prediction).
- Use bitmask to skip non-candidate positions: `if (bits >> i) & 1 != 0` instead of checking all 16 positions
- Process 2 chunks (32 bytes) at once with a single skip check

**Line extraction optimizations:**
- Currently uses scalar fallback after SIMD detect. Could do pure SIMD for common case.
- Skip line extraction entirely if lines_buf_cap == 0 (match-count-only mode)

**General:**
- `*restrict` on more parameters (NOT pt_buf/pt_i32)
- Reduce function call overhead (inline helpers manually if Ea doesn't inline)

## Available Eä Features

**SIMD types:** f32x4, f32x8, i32x4, i32x8, u8x16, u8x32, i8x16, i16x8, i16x16

**Vector dot operators:** `.+`, `.-`, `.*`, `./`, `.==`, `.!=`, `.<`, `.>`, `.<=`, `.>=`, `.&`, `.|`, `.^`, `.<<`, `.>>`

**Intrinsics:**
- Memory: `load`, `store`, `stream_store`, `load_masked`, `store_masked`, `prefetch(ptr, offset)`, `gather`, `scatter`
- Reduction: `reduce_add`, `reduce_min`, `reduce_max`
- Construction: `splat(scalar)`, `shuffle(vec, [indices])`, `select(mask, a, b)`
- Mask: `movemask(boolx16)` → i32 bitmask
- Math: `fma`, `sqrt`, `rsqrt`, `min`, `max`
- Conversion: `widen_u8_i32x4`, `widen_u8_f32x4`, etc.

**Scalar operators:** `+`, `-`, `*`, `/`, `%`, `&`, `|`, `^`, `<<`, `>>`, `==`, `!=`, `<`, `>`, `<=`, `>=`, `&&`, `||`, `!`

**NOT available:** `popcount`, `ctz`, `clz`, structs, heap allocation, pointer arithmetic

**Loop constructs:** `while`, `foreach`, `unroll(N)`

**Pointer annotations:** `*restrict`, `*restrict mut`, `*mut` (restrict = no aliasing, must be correct)

## Output Format

Your output MUST contain exactly two things:

1. A line starting with HYPOTHESIS: followed by what you are trying and why
2. The complete kernel.ea file wrapped in a markdown code fence tagged with ea

Do NOT omit the HYPOTHESIS line. Do NOT omit the code fence. Do NOT output partial kernels.
