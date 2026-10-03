"""
STEP 2: Self-attention, built from scratch, fed REAL GloVe embeddings from
your embeddings track. This directly resolves the unfinished business from
that track: proving 'mouse' gets a different vector depending on context.

The mechanism, in order:
  1. Every token's embedding gets projected into three roles via three
     learned weight matrices: Query (what am I looking for?), Key (what do
     I contain, for others to find?), Value (what do I actually offer once
     found?).
  2. Score every token against every other token: Q_i . K_j -- how much
     should token i attend to token j?
  3. Scale by sqrt(d_k) (keeps dot products from growing huge as dimension
     grows, which would push softmax into a near one-hot, near-zero-gradient
     regime) and softmax each token's scores into a probability distribution
     over all tokens (including itself).
  4. Output_i = weighted sum of ALL tokens' Value vectors, weighted by those
     softmax probabilities.

Step 4 is the whole point: output_i is a DIFFERENT function of the whole
sentence for every sentence, even for the same input word, because the
other tokens in the weighted sum are different. A static embedding lookup
structurally cannot do this -- it always returns the same row. Attention
structurally cannot avoid it -- the output always depends on context.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import gensim.downloader as api

np.random.seed(0)
wv = api.load("glove-wiki-gigaword-50")  # cached already from the embeddings track
EMBED_DIM = wv.vector_size


def softmax(x: np.ndarray) -> np.ndarray:
    # subtract max per row for numerical stability -- doesn't change the
    # result mathematically (softmax is shift-invariant) but avoids
    # overflow in exp() for large scores.
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def self_attention(X: np.ndarray, Wq, Wk, Wv):
    Q = X @ Wq                              # (n_tokens, d_k)
    K = X @ Wk                              # (n_tokens, d_k)
    V = X @ Wv                              # (n_tokens, d_v)
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)         # (n_tokens, n_tokens): every token vs every token
    weights = softmax(scores)               # each row sums to 1 -- a distribution over tokens
    output = weights @ V                    # (n_tokens, d_v): weighted blend of Value vectors
    return output, weights


# SAME random projection weights used for both sentences -- this isolates
# the effect to "different context" rather than "different weights."
Wq = np.random.randn(EMBED_DIM, EMBED_DIM) * 0.1
Wk = np.random.randn(EMBED_DIM, EMBED_DIM) * 0.1
Wv = np.random.randn(EMBED_DIM, EMBED_DIM) * 0.1

sentence_a = "i saw a mouse in the garden".split()
sentence_b = "i clicked the mouse with my hand".split()


def embed(tokens):
    return np.array([wv[t] for t in tokens])


X_a = embed(sentence_a)
X_b = embed(sentence_b)

out_a, weights_a = self_attention(X_a, Wq, Wk, Wv)
out_b, weights_b = self_attention(X_b, Wq, Wk, Wv)

idx_a = sentence_a.index("mouse")
idx_b = sentence_b.index("mouse")

mouse_static = wv["mouse"]
mouse_context_a = out_a[idx_a]
mouse_context_b = out_b[idx_b]


def cos(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


print(f"Sentence A: {' '.join(sentence_a)!r}")
print(f"Sentence B: {' '.join(sentence_b)!r}")
print()
print("Attention weights FOR 'mouse' (how much it attends to each word in its sentence):")
print(f"  A: {dict(zip(sentence_a, np.round(weights_a[idx_a], 3)))}")
print(f"  B: {dict(zip(sentence_b, np.round(weights_b[idx_b], 3)))}")
print()

print(f"cos(mouse_static, mouse_static)              = 1.000   (trivially -- same vector)")
print(f"cos(mouse_after_attention_A, mouse_static)   = {cos(mouse_context_a, mouse_static):+.4f}")
print(f"cos(mouse_after_attention_B, mouse_static)   = {cos(mouse_context_b, mouse_static):+.4f}")
print(f"cos(mouse_after_attention_A, mouse_after_attention_B) = {cos(mouse_context_a, mouse_context_b):+.4f}")
print()
print("The static GloVe vector for 'mouse' is IDENTICAL in both sentences (by")
print("construction -- it's a lookup table). After self-attention, 'mouse' gets a")
print("DIFFERENT output vector in each sentence, because it's now a weighted blend")
print("of that sentence's OTHER words' Value vectors -- and sentence A's other words")
print("('garden', 'saw') are different from sentence B's ('clicked', 'hand').")
print()
print("Honest caveat, consistent with the embeddings track: these are RANDOM,")
print("untrained Wq/Wk/Wv, so this doesn't yet prove the contextualization is")
print("semantically CORRECT (that 'mouse' correctly leans animal-ish in A and")
print("device-ish in B) -- only that attention structurally FORCES the output to")
print("depend on context, which a static embedding structurally cannot do.")
print()
print("Notice the attention weight dicts above are all clustered tightly around")
print("0.12-0.16 -- almost uniform (1/7 ~ 0.143 for a 7-token sentence), instead of")
print("sharply favoring one or two context words. That's WHY the A-vs-B similarity")
print("(0.92) is high rather than dramatically different: with small random Wq/Wk,")
print("Q.K scores stay close to 0, and softmax(near-zero scores) ~ uniform average.")
print("An almost-uniform average over similar small English sentences naturally")
print("comes out similar. Let's check that this really is a SCALE effect, not a")
print("fluke, by rerunning with larger-magnitude random projections.")
print()

Wq2, Wk2, Wv2 = (np.random.randn(EMBED_DIM, EMBED_DIM) for _ in range(3))
out_a2, weights_a2 = self_attention(X_a, Wq2 * 2.0, Wk2 * 2.0, Wv2)
out_b2, weights_b2 = self_attention(X_b, Wq2 * 2.0, Wk2 * 2.0, Wv2)
print("Same experiment, projection weights scaled up 20x (0.1 -> 2.0):")
print(f"  attention weights for 'mouse' in A: {dict(zip(sentence_a, np.round(weights_a2[idx_a], 3)))}")
print(f"  attention weights for 'mouse' in B: {dict(zip(sentence_b, np.round(weights_b2[idx_b], 3)))}")
print(f"  cos(mouse_after_attention_A, mouse_after_attention_B) = {cos(out_a2[idx_a], out_b2[idx_b]):+.4f}")
print()
print("With larger-scale projections the softmax sharpens -- attention actually")
print("concentrates on a couple of words instead of averaging everything -- and the")
print("A-vs-B similarity drops noticeably. This is exactly why the real formula")
print("divides by sqrt(d_k): unscaled Q.K grows with dimension and pushes softmax")
print("toward one-hot (near-zero gradient almost everywhere), while too-small scores")
print("collapse toward uniform averaging (which is what we saw first). Training")
print("finds weights in between -- sharp enough to be selective, stable enough to")
print("have a usable gradient -- and that's exactly what makes contextualization")
print("both meaningful AND learnable, not just structurally possible.")
print()
print("Real transformers train these projection matrices so the resulting attention")
print("weights actually pick out the contextually relevant words -- that's what")
print("training teaches, exactly like it taught GloVe which words are similar.")
