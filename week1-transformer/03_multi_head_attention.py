"""
STEP 3: Multi-head attention. Why isn't ONE attention computation enough?

Single-head attention produces exactly ONE softmax distribution per token
-- one weighted blend of the sentence. But a word can need to attend to
several DIFFERENT other words for several DIFFERENT reasons at once. E.g.
in "the tired dog that chased the cat slept", when building a
representation for "slept" you might want to attend to "dog" (who did the
sleeping -- a subject-verb relationship) AND to "tired" (why -- a semantic/
descriptive relationship) simultaneously. One softmax distribution has to
blend both needs into a single compromise. Multiple heads let DIFFERENT
subspaces of the embedding specialize on DIFFERENT relationship types,
computed in parallel, then combined.

Mechanically: instead of one big attention over the full embedding_dim,
split embedding_dim into num_heads equal slices, run independent
self-attention on each slice (each with its own Wq/Wk/Wv for that slice),
concatenate the per-head outputs back together, then apply one shared
output projection Wo to mix information across heads.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import gensim.downloader as api

np.random.seed(0)
wv = api.load("glove-wiki-gigaword-50")
EMBED_DIM = wv.vector_size  # 50
NUM_HEADS = 5
D_K = EMBED_DIM // NUM_HEADS  # 10 per head


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    weights = softmax(scores)
    return weights @ V, weights


def multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads):
    n_tokens, embed_dim = X.shape
    d_k = embed_dim // num_heads

    Q_full = X @ Wq   # (n_tokens, embed_dim)
    K_full = X @ Wk
    V_full = X @ Wv

    head_outputs = []
    all_weights = []
    for h in range(num_heads):
        sl = slice(h * d_k, (h + 1) * d_k)
        Q_h, K_h, V_h = Q_full[:, sl], K_full[:, sl], V_full[:, sl]
        out_h, w_h = attention(Q_h, K_h, V_h)
        head_outputs.append(out_h)
        all_weights.append(w_h)

    concatenated = np.concatenate(head_outputs, axis=-1)  # back to (n_tokens, embed_dim)
    output = concatenated @ Wo                              # mix information across heads
    return output, all_weights


Wq = np.random.randn(EMBED_DIM, EMBED_DIM) * 0.3
Wk = np.random.randn(EMBED_DIM, EMBED_DIM) * 0.3
Wv = np.random.randn(EMBED_DIM, EMBED_DIM) * 0.3
Wo = np.random.randn(EMBED_DIM, EMBED_DIM) * 0.3

sentence = "the tired dog that chased the cat slept".split()
X = np.array([wv[t] for t in sentence])

output, all_weights = multi_head_attention(X, Wq, Wk, Wv, Wo, NUM_HEADS)
slept_idx = sentence.index("slept")

print(f"Sentence: {' '.join(sentence)!r}")
print(f"Split {EMBED_DIM}-dim embeddings into {NUM_HEADS} heads of {D_K} dims each.")
print()
print(f"Attention weights FOR 'slept', one row per head (which words does 'slept'")
print(f"attend to, in each head's independent subspace):")
for h in range(NUM_HEADS):
    row = all_weights[h][slept_idx]
    top2_idx = np.argsort(row)[::-1][:2]
    top2 = [(sentence[i], round(float(row[i]), 3)) for i in top2_idx]
    print(f"  head {h}: top attended words = {top2}   full: {np.round(row, 2)}")
print()

# quantify how DIFFERENT the heads' attention patterns are from each other
print("Pairwise similarity between heads' attention patterns for 'slept'")
print("(1.0 = identical pattern, 0.0 = completely different focus):")
for h1 in range(NUM_HEADS):
    for h2 in range(h1 + 1, NUM_HEADS):
        w1, w2 = all_weights[h1][slept_idx], all_weights[h2][slept_idx]
        sim = np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2))
        print(f"  head {h1} vs head {h2}: {sim:.3f}")
print()

print("Honest caveat (same as step 2): these are random, untrained weights, so")
print("which specific words each head 'focuses on' isn't semantically meaningful")
print("yet. What IS real and structural: each head operates on a genuinely")
print("different 10-dim slice of the embedding, so they see different projections")
print("of the same tokens and can mathematically land on different attention")
print("patterns -- which the similarity numbers above confirm (not all 1.0, i.e.")
print("not identical). Training is what teaches specific heads to specialize on")
print("specific relationship types -- real interpretability research on trained")
print("GPT-style models finds heads that reliably track things like 'attend to the")
print("previous occurrence of this exact token' or 'attend to the subject of the")
print("current clause,' completely automatically, purely from being useful for")
print("next-token prediction during training.")
