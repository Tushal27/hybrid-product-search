"""
WEEK 3, STEP 4: Quantization -- storing weights in fewer bits than the
32-bit floats they were trained/computed in, to shrink memory footprint
and speed up inference, at the cost of some numerical precision.

The core idea (this is genuinely how it works, not a simplification):
  1. Find the weight matrix's min and max value.
  2. Map that [min, max] float range onto the full range of a small
     integer type (0..255 for 8-bit, 0..15 for 4-bit) -- this defines a
     SCALE (how much real value one integer step represents) and a
     ZERO_POINT (what float value integer 0 represents).
  3. Store the small integers instead of the floats. To USE the weight,
     DEQUANTIZE: integer * scale + zero_point recovers an approximation
     of the original float.

This script does three concrete things: (a) quantize real weight matrices
to int8 AND int4 (with actual bit-packing -- two 4-bit values really
crammed into one byte, not just "imagine 4 bits"), (b) measure the real
memory savings in bytes, and (c) plug the dequantized weights back into
an actual forward pass and measure how much the OUTPUT changes -- proving
the accuracy cost is small, not just asserting it.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

np.random.seed(0)


# ---------------------------------- quantize / dequantize ----------------------------------
def quantize(W, num_bits):
    """Per-tensor min-max quantization to an unsigned integer of num_bits."""
    levels = 2 ** num_bits - 1  # 255 for 8-bit, 15 for 4-bit
    w_min, w_max = W.min(), W.max()
    scale = (w_max - w_min) / levels
    zero_point = w_min
    q = np.round((W - zero_point) / scale).astype(np.uint8)
    return q, scale, zero_point


def dequantize(q, scale, zero_point):
    return q.astype(np.float32) * scale + zero_point


def pack_int4(q4_flat):
    """Actually cram two 4-bit values into one real byte -- this is what makes
    int4 storage genuinely 8x smaller than fp32, not just a labeling exercise."""
    if len(q4_flat) % 2 != 0:
        q4_flat = np.append(q4_flat, 0)
    high, low = q4_flat[0::2], q4_flat[1::2]
    return ((high << 4) | low).astype(np.uint8)


def unpack_int4(packed, num_values):
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    out = np.empty(packed.size * 2, dtype=np.uint8)
    out[0::2], out[1::2] = high, low
    return out[:num_values]


# ---------------------------------- 1. quantize a real weight matrix ----------------------------------
print("=" * 78)
print("1. QUANTIZE A REAL WEIGHT MATRIX TO INT8 AND INT4")
print("=" * 78)
EMBED_DIM = 64
rng = np.random.default_rng(0)
W = (rng.standard_normal((EMBED_DIM, EMBED_DIM)) * 0.3).astype(np.float32)  # stand-in for a real weight matrix
print(f"Original weight matrix: shape {W.shape}, dtype {W.dtype}")
print(f"  range: [{W.min():.4f}, {W.max():.4f}]")
print()

q8, scale8, zp8 = quantize(W, num_bits=8)
W_dq8 = dequantize(q8, scale8, zp8)
err8 = np.abs(W - W_dq8)
print(f"INT8: scale={scale8:.6f}, zero_point={zp8:.4f}")
print(f"  reconstruction error: mean={err8.mean():.6f}, max={err8.max():.6f}  "
      f"(vs weight range width {W.max()-W.min():.4f})")

q4, scale4, zp4 = quantize(W, num_bits=4)
W_dq4 = dequantize(q4, scale4, zp4)
err4 = np.abs(W - W_dq4)
print(f"INT4: scale={scale4:.6f}, zero_point={zp4:.4f}")
print(f"  reconstruction error: mean={err4.mean():.6f}, max={err4.max():.6f}  "
      f"({err4.mean()/err8.mean():.1f}x worse than int8 -- only 16 distinct integer")
print(f"  levels total, vs int8's 256, to represent this whole weight range)")
print()

# prove the bit-packing round-trips correctly
packed = pack_int4(q4.flatten())
unpacked = unpack_int4(packed, q4.size).reshape(q4.shape)
print(f"int4 bit-packing round-trip check: unpack(pack(q4)) == q4 -> {np.array_equal(unpacked, q4)}")
print()


# ---------------------------------- 2. real memory savings ----------------------------------
print("=" * 78)
print("2. ACTUAL MEMORY FOOTPRINT, NOT JUST A RATIO")
print("=" * 78)
num_params = W.size
fp32_bytes = num_params * 4
int8_bytes = num_params * 1
int4_bytes = packed.nbytes  # the REAL packed byte array from above, not a computed estimate

print(f"{num_params} parameters in this one matrix:")
print(f"  fp32 : {fp32_bytes:>6} bytes")
print(f"  int8 : {int8_bytes:>6} bytes  ({fp32_bytes/int8_bytes:.1f}x smaller)")
print(f"  int4 : {int4_bytes:>6} bytes  ({fp32_bytes/int4_bytes:.1f}x smaller, ACTUALLY packed, "
      f"not estimated)")
print()

for param_count, label in [(7_000_000_000, "a 7B-parameter model"), (70_000_000_000, "a 70B-parameter model")]:
    fp32_gb = param_count * 4 / 1024**3
    int8_gb = param_count * 1 / 1024**3
    int4_gb = param_count * 0.5 / 1024**3
    print(f"  {label}: fp32={fp32_gb:>6.1f} GB   int8={int8_gb:>6.1f} GB   int4={int4_gb:>6.1f} GB")
print("(this is the actual, literal reason int4/int8 quantized models -- the GGUF")
print("files in step 5 -- are what let a 7-70B model run on a consumer laptop's")
print("RAM/VRAM at all, where the original fp32/fp16 weights simply wouldn't fit.)")
print()


# ---------------------------------- 3. does it actually still work? ----------------------------------
print("=" * 78)
print("3. PLUG DEQUANTIZED WEIGHTS INTO A REAL FORWARD PASS -- DOES OUTPUT SURVIVE?")
print("=" * 78)


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads, mask=None):
    n_tokens, embed_dim = X.shape
    d_k = embed_dim // num_heads
    Q_full, K_full, V_full = X @ Wq, X @ Wk, X @ Wv
    head_outputs = []
    for h in range(num_heads):
        sl = slice(h * d_k, (h + 1) * d_k)
        Q_h, K_h, V_h = Q_full[:, sl], K_full[:, sl], V_full[:, sl]
        scores = Q_h @ K_h.T / np.sqrt(d_k)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        head_outputs.append(softmax(scores) @ V_h)
    return np.concatenate(head_outputs, axis=-1) @ Wo


NUM_HEADS = 8
n_tokens = 6
X = rng.standard_normal((n_tokens, EMBED_DIM)).astype(np.float32)
Wk_ = (rng.standard_normal((EMBED_DIM, EMBED_DIM)) * 0.3).astype(np.float32)
Wv_ = (rng.standard_normal((EMBED_DIM, EMBED_DIM)) * 0.3).astype(np.float32)
Wo_ = (rng.standard_normal((EMBED_DIM, EMBED_DIM)) * 0.3).astype(np.float32)
mask = np.tril(np.ones((n_tokens, n_tokens), dtype=bool))

out_fp32 = multi_head_attention(X, W, Wk_, Wv_, Wo_, NUM_HEADS, mask=mask)

# quantize ONLY Wq (the matrix from step 1) to int8, then int4, and rerun
out_int8 = multi_head_attention(X, W_dq8, Wk_, Wv_, Wo_, NUM_HEADS, mask=mask)
out_int4 = multi_head_attention(X, W_dq4, Wk_, Wv_, Wo_, NUM_HEADS, mask=mask)

diff8 = np.abs(out_fp32 - out_int8)
diff4 = np.abs(out_fp32 - out_int4)
rel8 = diff8.mean() / np.abs(out_fp32).mean()
rel4 = diff4.mean() / np.abs(out_fp32).mean()

print(f"Attention output with original fp32 Wq  : mean magnitude {np.abs(out_fp32).mean():.4f}")
print(f"Attention output with int8-quantized Wq : mean abs diff from fp32 = {diff8.mean():.6f}  "
      f"({rel8:.2%} relative change)")
print(f"Attention output with int4-quantized Wq : mean abs diff from fp32 = {diff4.mean():.6f}  "
      f"({rel4:.2%} relative change)")
print()
print("int8 barely moves the output at all -- this is why int8 quantization is often")
print("described as 'nearly free' accuracy-wise. int4's error is visibly bigger (only")
print("16 possible values per weight!) -- real int4 quantization schemes (like the")
print("ones GGUF uses, step 5) use tricks this simple demo skips -- per-CHANNEL scales")
print("instead of one scale for the whole matrix, mixed precision for sensitive")
print("layers, etc. -- specifically to close this gap back down.")
