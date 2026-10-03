"""
WEEK 3, STEP 3: Prompt caching -- the same KV-cache idea from step 2, but
applied ACROSS separate requests instead of within one generation loop.

Step 2's KV cache avoided recomputing old tokens' Key/Value vectors
between generation steps of ONE request. Prompt caching asks a different
question: what if MANY DIFFERENT requests all start with the exact same
long prefix? A system prompt, a big block of tool definitions, a long
document the user keeps asking different questions about -- that prefix
is IDENTICAL every time, token for token. Its K/V vectors would come out
byte-for-byte identical every single request too (same tokens, same
positions, same weights). Recomputing them per-request is pure waste --
so real APIs (this is literally what Anthropic's and OpenAI's "prompt
caching" features do) compute that shared prefix's KV cache ONCE, keep it
around, and every new request that shares the prefix just continues from
it -- only paying full compute for whatever's actually NEW and different.

This script: build one shared prefix's KV cache once, reuse it for
several DIFFERENT suffixes (as if from different users/requests), verify
the result is numerically identical to computing the whole thing (prefix
+ suffix) from scratch every time, then quantify the compute saved across
many requests sharing that prefix.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

np.random.seed(0)


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


class CachedCausalAttention:
    """Same math as week3-memory-performance/02_kv_cache.py, generalized to
    accept a CHUNK of new tokens at once (a whole suffix), not just one
    token -- prompt caching resumes with a full new suffix, not one at a time."""

    def __init__(self, embed_dim, seed=0):
        rng = np.random.default_rng(seed)
        scale = 0.3
        self.Wq = rng.standard_normal((embed_dim, embed_dim)) * scale
        self.Wk = rng.standard_normal((embed_dim, embed_dim)) * scale
        self.Wv = rng.standard_normal((embed_dim, embed_dim)) * scale
        self.d_k = embed_dim

    def forward_from_scratch(self, X):
        """Full, uncached forward pass over the ENTIRE sequence -- the baseline
        every real request would cost with no prompt caching at all."""
        Q, K, V = X @ self.Wq, X @ self.Wk, X @ self.Wv
        n = X.shape[0]
        mask = np.tril(np.ones((n, n), dtype=bool))
        scores = np.where(mask, Q @ K.T / np.sqrt(self.d_k), -np.inf)
        return softmax(scores) @ V

    def compute_prefix_cache(self, prefix):
        """Run the shared prefix ONCE, return its K,V -- this is the thing that
        gets cached and reused across every request sharing this prefix."""
        Q, K, V = prefix @ self.Wq, prefix @ self.Wk, prefix @ self.Wv
        n = prefix.shape[0]
        mask = np.tril(np.ones((n, n), dtype=bool))
        scores = np.where(mask, Q @ K.T / np.sqrt(self.d_k), -np.inf)
        return softmax(scores) @ V, K, V

    def continue_from_cache(self, suffix, K_cache, V_cache):
        """Only the NEW suffix tokens get projected to Q/K/V. Its queries attend
        over cached prefix K/V (freely -- prefix is all 'in the past') PLUS the
        suffix's own K/V (causally masked among themselves)."""
        m = suffix.shape[0]
        n_old = K_cache.shape[0]
        Q_new, K_new, V_new = suffix @ self.Wq, suffix @ self.Wk, suffix @ self.Wv
        K_full = np.vstack([K_cache, K_new])
        V_full = np.vstack([V_cache, V_new])

        mask = np.zeros((m, n_old + m), dtype=bool)
        mask[:, :n_old] = True                       # every suffix token sees the WHOLE prefix
        mask[:, n_old:] = np.tril(np.ones((m, m), dtype=bool))  # + causal among suffix tokens

        scores = np.where(mask, Q_new @ K_full.T / np.sqrt(self.d_k), -np.inf)
        return softmax(scores) @ V_full


EMBED_DIM = 16
attn = CachedCausalAttention(EMBED_DIM)

rng = np.random.default_rng(1)
PREFIX_LEN = 20   # stands in for a long system prompt / tool definitions / shared document
prefix = rng.standard_normal((PREFIX_LEN, EMBED_DIM))

suffixes = [rng.standard_normal((5, EMBED_DIM)) for _ in range(3)]  # 3 different "requests"

print("=" * 78)
print("1. COMPUTE THE SHARED PREFIX'S KV CACHE ONCE")
print("=" * 78)
prefix_output, K_cache, V_cache = attn.compute_prefix_cache(prefix)
print(f"Prefix: {PREFIX_LEN} tokens (e.g. a long system prompt). Computed its K,V once:")
print(f"  K_cache shape: {K_cache.shape}, V_cache shape: {V_cache.shape}")
print("This is exactly what a 'cache_control' / automatic prompt-caching hit means")
print("in a real API: this prefix's K,V already exist, don't redo this work.")
print()

print("=" * 78)
print("2. REUSE THAT SAME CACHE FOR 3 DIFFERENT REQUESTS' SUFFIXES")
print("=" * 78)
for i, suffix in enumerate(suffixes):
    cached_out = attn.continue_from_cache(suffix, K_cache, V_cache)

    # ground truth: recompute prefix+suffix TOGETHER from scratch, take only the
    # suffix's rows -- this is what NO prompt caching would cost every single time
    full_sequence = np.vstack([prefix, suffix])
    full_out = attn.forward_from_scratch(full_sequence)
    suffix_out_from_scratch = full_out[PREFIX_LEN:]

    max_diff = np.max(np.abs(cached_out - suffix_out_from_scratch))
    print(f"  request {i+1}: suffix={suffix.shape[0]} new tokens -> "
          f"max|cached - from_scratch| = {max_diff:.2e}   MATCH: {max_diff < 1e-10}")
print()
print("Every request's suffix output is numerically IDENTICAL whether the prefix")
print("was recomputed from scratch or reused from cache -- same guarantee as step 2's")
print("KV cache, just spanning separate requests instead of separate generation steps.")
print()


# ---------------------------------- 3. how much compute does this actually save ----------------------------------
print("=" * 78)
print("3. COMPUTE SAVED, AT A REALISTIC SCALE")
print("=" * 78)
print("Attention cost (~query-key comparisons) per request:")
print("  WITHOUT prompt caching: recompute prefix+suffix together -> ~(prefix+suffix)^2")
print("  WITH    prompt caching: prefix cost paid ONCE ever, then each request only")
print("                          costs ~suffix * (prefix+suffix)  (new queries, but")
print("                          against the full old+new key set)")
print()

for prefix_len, suffix_len, num_requests in [(2_000, 50, 100), (8_000, 100, 1_000)]:
    no_cache_total = num_requests * (prefix_len + suffix_len) ** 2
    with_cache_total = prefix_len ** 2 + num_requests * (suffix_len * (prefix_len + suffix_len))
    print(f"  prefix={prefix_len:,} tokens, suffix={suffix_len} tokens, "
          f"{num_requests:,} requests sharing this prefix:")
    print(f"    without prompt caching : {no_cache_total:>18,}  (pays for the FULL prefix, every request)")
    print(f"    with    prompt caching : {with_cache_total:>18,}  (pays for the prefix ONCE, ever)")
    print(f"    -> {no_cache_total / with_cache_total:.1f}x less attention work")
    print()

print("This is exactly why a long, STABLE system prompt or tool-definition block is")
print("cheap in practice despite being huge, as long as it's reused across many")
print("calls unchanged -- and why changing even one token near the START of that")
print("shared prefix invalidates the whole cache for everything after it (the K,V")
print("for every later token was computed attending to the OLD version of what came")
print("before it -- change the prefix, and all of that is stale).")
