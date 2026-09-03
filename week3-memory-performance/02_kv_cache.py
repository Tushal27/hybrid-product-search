"""
WEEK 3, STEP 2: KV caching -- why real inference servers don't recompute
the whole sequence from scratch at every generated token.

This picks up right where week 2's inference pipeline left off: that
pipeline works, but at every generation step it reprocesses the ENTIRE
sequence from position 0. This step is the first "memory & performance"
optimization on top of a correct pipeline -- same category as context
windows, prompt caching, and quantization (the rest of week 3).

NAIVE autoregressive generation: to generate token n+1, run the WHOLE
sequence (tokens 0..n) through the model again. Wasteful -- because of the
causal mask, token i's Key and Value vectors depend ONLY on tokens 0..i,
never on anything after. So K and V for positions 0..n-1 are EXACTLY the
same values they were on the previous step. Recomputing them is pure
duplicated work.

KV CACHE: store every layer's K and V vectors as they're computed. On the
next step, only run the ONE new token through Q/K/V projections, append
its new K,V to the cache, and attend the new token's Q against the FULL
cached K,V (old + new). Skips recomputing every previous position's K,V.

This script proves two things: (1) the cached approach produces IDENTICAL
output to the naive full-recompute approach (numerically, not just
"close"), and (2) it does asymptotically less matmul work as the sequence
grows -- shown here as a growing gap in FLOPs, since Python-level
overhead on tiny toy sizes would otherwise hide the real speedup you'd see
in a production server.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

np.random.seed(0)


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


class CausalSelfAttention:
    """Single-head causal self-attention, small enough to reason about by hand."""

    def __init__(self, embed_dim, seed=0):
        rng = np.random.default_rng(seed)
        scale = 0.3
        self.Wq = rng.standard_normal((embed_dim, embed_dim)) * scale
        self.Wk = rng.standard_normal((embed_dim, embed_dim)) * scale
        self.Wv = rng.standard_normal((embed_dim, embed_dim)) * scale
        self.d_k = embed_dim

    def forward_naive(self, X):
        """Recompute Q,K,V for the ENTIRE sequence X every call."""
        Q, K, V = X @ self.Wq, X @ self.Wk, X @ self.Wv
        n = X.shape[0]
        mask = np.tril(np.ones((n, n), dtype=bool))
        scores = Q @ K.T / np.sqrt(self.d_k)
        scores = np.where(mask, scores, -np.inf)
        weights = softmax(scores)
        return weights @ V, K, V

    def forward_one_step_cached(self, x_new, K_cache, V_cache):
        """
        x_new: just the ONE new token's embedding, shape (embed_dim,).
        K_cache, V_cache: (n_so_far, embed_dim) from all PREVIOUS steps.
        Returns this new token's output, plus the updated (appended) cache.
        """
        q_new = x_new @ self.Wq          # (embed_dim,) -- only ONE new Q, not the whole sequence
        k_new = x_new @ self.Wk
        v_new = x_new @ self.Wv
        K_full = np.vstack([K_cache, k_new]) if K_cache.size else k_new[None, :]
        V_full = np.vstack([V_cache, v_new]) if V_cache.size else v_new[None, :]
        scores = (q_new @ K_full.T) / np.sqrt(self.d_k)   # attend against ALL keys so far, no mask needed --
        weights = softmax(scores)                          # there's nothing "in the future" left to hide
        out_new = weights @ V_full
        return out_new, K_full, V_full


EMBED_DIM = 16
attn = CausalSelfAttention(EMBED_DIM)

rng = np.random.default_rng(1)
full_sequence = rng.standard_normal((8, EMBED_DIM))  # pretend these are 8 tokens' embeddings

# ---------- ground truth: naive full recompute at each step ----------
naive_outputs = []
for t in range(1, len(full_sequence) + 1):
    out, _, _ = attn.forward_naive(full_sequence[:t])
    naive_outputs.append(out[-1])          # only the newest position's output matters at generation time
naive_outputs = np.array(naive_outputs)

# ---------- cached: only process the newest token each step ----------
K_cache = np.empty((0, EMBED_DIM))
V_cache = np.empty((0, EMBED_DIM))
cached_outputs = []
for t in range(len(full_sequence)):
    out_new, K_cache, V_cache = attn.forward_one_step_cached(full_sequence[t], K_cache, V_cache)
    cached_outputs.append(out_new)
cached_outputs = np.array(cached_outputs)

max_diff = np.max(np.abs(naive_outputs - cached_outputs))
print(f"Sequence length: {len(full_sequence)} tokens, embed_dim={EMBED_DIM}")
print(f"max|naive_output - cached_output| = {max_diff:.2e}")
print(f"MATCH: {max_diff < 1e-10}")
print()
print("Every generated token's output is numerically IDENTICAL whether we")
print("recompute the whole sequence from scratch or reuse a growing KV cache --")
print("the cache is a pure speed optimization, changes zero numbers.")
print()

# ---------- how much work does each approach actually do? ----------
print("=" * 70)
print("QK^T + weights@V matmul cost per generation step (in 'token-vector ops'):")
print(f"{'step':>5}  {'naive (recompute all)':>24}  {'cached (new token only)':>26}")
naive_total, cached_total = 0, 0
for t in range(1, 33):
    # naive: attending t queries against t keys costs O(t^2)
    naive_cost = t * t
    # cached: attending 1 new query against t keys costs O(t)
    cached_cost = t
    naive_total += naive_cost
    cached_total += cached_cost
    if t <= 8 or t in (16, 32):
        print(f"{t:>5}  {naive_cost:>24}  {cached_cost:>26}")
print()
print(f"Total cost generating 32 tokens: naive={naive_total:,}   cached={cached_total:,}")
print(f"Naive does {naive_total / cached_total:.1f}x more attention work than cached")
print("for a 32-token generation, and that ratio grows LINEARLY with sequence")
print("length -- this is exactly why every real LLM inference server (vLLM,")
print("TensorRT-LLM, llama.cpp, etc.) is built around a KV cache, and why VRAM")
print("usage during generation is dominated by 'how big is the KV cache', not")
print("'how big are the model weights'.")
