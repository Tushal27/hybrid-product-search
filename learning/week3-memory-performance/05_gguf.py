"""
WEEK 3, STEP 5: GGUF -- what a downloaded model file actually IS.

GGUF (used by llama.cpp, Ollama, LM Studio, etc.) is a FILE FORMAT, not an
algorithm -- it's the container that lets you download one file and get
everything needed to run a model locally: metadata (architecture,
tokenizer vocab, hyperparameters) AND the (usually quantized, step 4)
weight tensors, laid out so the runtime can memory-map straight from disk
without a separate loading/parsing step.

This script doesn't reimplement the real GGUF spec (that's a real binary
format with its own typed KV encoding, alignment rules, etc.) -- instead
it builds a genuinely working MINI version with the same conceptual
structure, so "model file" stops being a black box:

    [HEADER: magic bytes, version]
    [METADATA: architecture name, vocab_size, embed_dim, ... -- the stuff
               you'd need to even KNOW HOW to construct the right model
               class before you can load the weights into it]
    [TENSOR INDEX: for each weight matrix -- its name, shape, quantization
               type, and WHERE in the file its raw bytes start]
    [TENSOR DATA: the actual quantized weight bytes, back to back]

Then it writes a real file to disk, in fp32 and in quantized form, and
compares REAL file sizes with os.path.getsize -- not estimates.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import json
import struct
import numpy as np

np.random.seed(0)

MAGIC = b"MGUF"     # our mini-format's magic bytes (real GGUF's is "GGUF")
VERSION = 1


def quantize_int8(W):
    w_min, w_max = W.min(), W.max()
    scale = (w_max - w_min) / 255
    zero_point = w_min
    q = np.round((W - zero_point) / scale).astype(np.uint8)
    return q, float(scale), float(zero_point)


def dequantize_int8(q, scale, zero_point):
    return q.astype(np.float32) * scale + zero_point


def save_model_file(path, metadata: dict, tensors: dict, quantize=False):
    """tensors: name -> np.ndarray (float32). Writes header + metadata +
    tensor index + raw tensor bytes, mirroring GGUF's actual section layout."""
    tensor_index = []
    data_blob = bytearray()

    for name, W in tensors.items():
        if quantize:
            q, scale, zero_point = quantize_int8(W)
            raw_bytes = q.tobytes()
            dtype_tag = "int8"
            quant_params = {"scale": scale, "zero_point": zero_point}
        else:
            raw_bytes = W.astype(np.float32).tobytes()
            dtype_tag = "float32"
            quant_params = {}

        tensor_index.append({
            "name": name,
            "shape": list(W.shape),
            "dtype": dtype_tag,
            "quant_params": quant_params,
            "offset": len(data_blob),
            "nbytes": len(raw_bytes),
        })
        data_blob.extend(raw_bytes)

    metadata_bytes = json.dumps(metadata).encode("utf-8")
    index_bytes = json.dumps(tensor_index).encode("utf-8")

    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", len(metadata_bytes)))
        f.write(metadata_bytes)
        f.write(struct.pack("<I", len(index_bytes)))
        f.write(index_bytes)
        f.write(data_blob)


def load_model_file(path):
    with open(path, "rb") as f:
        magic = f.read(4)
        assert magic == MAGIC, f"not a MGUF file (got {magic!r})"
        version, = struct.unpack("<I", f.read(4))

        meta_len, = struct.unpack("<I", f.read(4))
        metadata = json.loads(f.read(meta_len))

        index_len, = struct.unpack("<I", f.read(4))
        tensor_index = json.loads(f.read(index_len))

        data_start = f.tell()
        tensors = {}
        for entry in tensor_index:
            f.seek(data_start + entry["offset"])
            raw = f.read(entry["nbytes"])
            if entry["dtype"] == "float32":
                arr = np.frombuffer(raw, dtype=np.float32).reshape(entry["shape"])
            else:
                q = np.frombuffer(raw, dtype=np.uint8).reshape(entry["shape"])
                arr = dequantize_int8(q, entry["quant_params"]["scale"],
                                       entry["quant_params"]["zero_point"])
            tensors[entry["name"]] = arr
    return version, metadata, tensors


# ---------------------------------- build a tiny "model" and save it ----------------------------------
print("=" * 78)
print("1. BUILD A TINY MODEL'S WEIGHTS, SAVE THEM AS TWO REAL FILES")
print("=" * 78)
VOCAB_SIZE, EMBED_DIM = 200, 32
rng = np.random.default_rng(0)
tensors = {
    "token_embed.weight": rng.standard_normal((VOCAB_SIZE, EMBED_DIM)).astype(np.float32) * 0.02,
    "block0.attn.Wq": rng.standard_normal((EMBED_DIM, EMBED_DIM)).astype(np.float32) * 0.3,
    "block0.attn.Wk": rng.standard_normal((EMBED_DIM, EMBED_DIM)).astype(np.float32) * 0.3,
    "block0.ff.W1": rng.standard_normal((EMBED_DIM, EMBED_DIM * 4)).astype(np.float32) * 0.3,
    "lm_head.weight": rng.standard_normal((EMBED_DIM, VOCAB_SIZE)).astype(np.float32) * 0.3,
}
metadata = {
    "architecture": "MiniGPT",
    "vocab_size": VOCAB_SIZE,
    "embed_dim": EMBED_DIM,
    "num_layers": 1,
    "tokenizer": "byte-level-bpe",   # a real GGUF file embeds the tokenizer too --
                                       # this is exactly why one file is enough to run it
}

out_dir = os.path.dirname(os.path.abspath(__file__))
fp32_path = os.path.join(out_dir, "_demo_model.f32.mguf")
quant_path = os.path.join(out_dir, "_demo_model.q8.mguf")

save_model_file(fp32_path, metadata, tensors, quantize=False)
save_model_file(quant_path, metadata, tensors, quantize=True)

fp32_size = os.path.getsize(fp32_path)
quant_size = os.path.getsize(quant_path)
total_params = sum(t.size for t in tensors.values())
print(f"Model: {total_params:,} total parameters across {len(tensors)} tensors")
print(f"Saved to disk (REAL files, REAL os.path.getsize):")
print(f"  {os.path.basename(fp32_path)}  : {fp32_size:>8,} bytes")
print(f"  {os.path.basename(quant_path)} : {quant_size:>8,} bytes  "
      f"({fp32_size/quant_size:.2f}x smaller on disk)")
print()

with open(fp32_path, "rb") as f:
    header_preview = f.read(64)
print(f"First 64 bytes of the fp32 file (magic + version + start of metadata JSON):")
print(f"  {header_preview}")
print()

# ---------------------------------- load it back and verify ----------------------------------
print("=" * 78)
print("2. LOAD IT BACK -- METADATA TELLS YOU HOW TO EVEN BUILD THE RIGHT MODEL")
print("=" * 78)
version, loaded_metadata, loaded_tensors = load_model_file(quant_path)
print(f"File version: {version}")
print(f"Metadata (this is what a real loader reads FIRST, before touching any")
print(f"weight bytes, to know what architecture/shapes to even expect):")
for k, v in loaded_metadata.items():
    print(f"    {k}: {v}")
print()

print("Tensor-by-tensor reconstruction error (quantized file vs the original fp32):")
for name, original in tensors.items():
    loaded = loaded_tensors[name]
    max_diff = np.max(np.abs(original - loaded))
    print(f"  {name:<20} shape={str(original.shape):<14} max diff = {max_diff:.6f}")
print()

os.remove(fp32_path)
os.remove(quant_path)
print("(demo files cleaned up)")
print()

print("=" * 78)
print("3. WHAT REAL GGUF FILENAMES ACTUALLY MEAN")
print("=" * 78)
print("You'll see filenames like: llama-3-8b.Q4_K_M.gguf, mistral-7b.Q8_0.gguf,")
print("phi-3-mini.F16.gguf. That's not a random SKU -- it directly maps to the")
print("quantization scheme from step 4:")
print("  F16 / F32   : not quantized -- full/half precision, biggest, most accurate")
print("  Q8_0        : 8-bit, one scale per block of weights -- ~4x smaller than F32,")
print("                barely any quality loss (matches this script's int8 result)")
print("  Q4_0 / Q4_K : 4-bit variants -- ~8x smaller than F32. 'K' variants (Q4_K_M,")
print("                Q4_K_S, etc.) use SMARTER per-block scales and mixed precision")
print("                for sensitive layers, specifically to claw back the accuracy")
print("                gap step 4 showed plain int4 has -- more sophisticated than")
print("                this demo's one-scale-for-the-whole-matrix approach.")
print()
print("Choosing a GGUF file to download is choosing a point on exactly the memory-vs-")
print("accuracy tradeoff curve step 4 measured: bigger number after Q = more bits =")
print("bigger file + closer to the original model's real quality.")
