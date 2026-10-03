"""
STEP 5: Encoder vs decoder, and the causal mask that makes GPT a
"decoder-only" model.

ENCODER (e.g. BERT): every token attends to EVERY other token in the
sequence, including ones that come after it. Good for understanding a
whole, already-complete piece of text (classification, embeddings for
search, etc.) -- there's no "future" to leak, the whole input is given
upfront.

DECODER (e.g. GPT): trained to predict the NEXT token given only the
tokens so FAR. If we let token i attend to token i+1, i+2, ... during
training, the model could just copy the answer directly from the future
token it's supposed to be predicting -- trivially "cheating," and useless
at actual generation time when those future tokens don't exist yet. The
fix: a CAUSAL MASK that forces token i's attention weights to be exactly
zero for every position j > i, before the softmax even runs.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import gensim.downloader as api

np.random.seed(0)
wv = api.load("glove-wiki-gigaword-50")
EMBED_DIM = wv.vector_size


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def self_attention(X, Wq, Wk, Wv, mask=None):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)   # -inf -> exp(-inf) = 0 after softmax
    weights = softmax(scores)
    return weights @ V, weights


Wq = np.random.randn(EMBED_DIM, EMBED_DIM) * 0.3
Wk = np.random.randn(EMBED_DIM, EMBED_DIM) * 0.3
Wv = np.random.randn(EMBED_DIM, EMBED_DIM) * 0.3

sentence = "the cat sat on the mat".split()
X = np.array([wv[t] for t in sentence])
n = len(sentence)

# encoder-style: every token can see every token (no mask)
_, weights_encoder = self_attention(X, Wq, Wk, Wv, mask=None)

# decoder-style: token i can only see tokens 0..i (lower triangular mask)
causal_mask = np.tril(np.ones((n, n), dtype=bool))
_, weights_decoder = self_attention(X, Wq, Wk, Wv, mask=causal_mask)

print(f"Sentence: {' '.join(sentence)!r}")
print()
print("Causal mask (True = allowed to attend, False = blocked):")
print("        " + "  ".join(f"{w:>5}" for w in sentence))
for i, row in enumerate(causal_mask):
    print(f"{sentence[i]:>7} " + "  ".join(f"{str(v):>5}" for v in row))
print()

print("ENCODER-style attention weights (bidirectional -- note nonzero values")
print("in the upper-right, meaning early tokens attend to LATER tokens too):")
print("        " + "  ".join(f"{w:>6}" for w in sentence))
for i, row in enumerate(weights_encoder):
    print(f"{sentence[i]:>7} " + "  ".join(f"{v:6.3f}" for v in row))
print()

print("DECODER-style attention weights (causal -- upper-right is EXACTLY 0.000,")
print("every row's probabilities are redistributed only over positions <= i):")
print("        " + "  ".join(f"{w:>6}" for w in sentence))
for i, row in enumerate(weights_decoder):
    print(f"{sentence[i]:>7} " + "  ".join(f"{v:6.3f}" for v in row))
print()

upper_triangle_mass = np.sum(weights_decoder[np.triu_indices(n, k=1)])
print(f"Total attention probability mass in the upper triangle (future positions): "
      f"{upper_triangle_mass:.10f}")
print("Exactly zero -- not 'very small', not 'approximately' -- because those")
print("positions were set to -inf before softmax, and exp(-inf) is exactly 0.0.")
print()
print("This is precisely why GPT is called 'decoder-only': it's built entirely")
print("from this causal-masked attention (stacked in many layers, each followed")
print("by a feedforward network -- step 6 puts these pieces together), trained")
print("to predict token i+1 from tokens 0..i, and at generation time it simply")
print("keeps applying that same rule one new token at a time.")
