"""
STEP 4: Expose the flaw in what we've built so far, then fix it.

Self-attention (steps 2-3) computes output_i from Q_i and the SET of all
(K_j, V_j) pairs -- nothing in the formula ever looks at WHERE in the
sequence token j sits, only at its content. This means attention is
"permutation equivariant": permute the input tokens' order and the output
just gets permuted the same way, with NO other change. Concretely: a
token's contextual representation is IDENTICAL regardless of where the
other tokens sit relative to it, as long as the same multiset of tokens
is present.

That's a real problem: "the dog bit the man" and "the man bit the dog"
contain the EXACT SAME multiset of words. Without position information,
attention literally cannot tell who bit whom.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import numpy as np
import gensim.downloader as api
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(0)
wv = api.load("glove-wiki-gigaword-50")
EMBED_DIM = wv.vector_size


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def self_attention(X, Wq, Wk, Wv):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    scores = Q @ K.T / np.sqrt(Q.shape[-1])
    weights = softmax(scores)
    return weights @ V


Wq = np.random.randn(EMBED_DIM, EMBED_DIM) * 0.3
Wk = np.random.randn(EMBED_DIM, EMBED_DIM) * 0.3
Wv = np.random.randn(EMBED_DIM, EMBED_DIM) * 0.3

sentence_1 = "the dog bit the man".split()
sentence_2 = "the man bit the dog".split()   # same multiset of words, swapped roles

X1 = np.array([wv[t] for t in sentence_1])
X2 = np.array([wv[t] for t in sentence_2])

out1 = self_attention(X1, Wq, Wk, Wv)
out2 = self_attention(X2, Wq, Wk, Wv)

dog_in_1 = out1[sentence_1.index("dog")]
dog_in_2 = out2[sentence_2.index("dog")]

print(f"Sentence 1: {' '.join(sentence_1)!r}  (dog is the SUBJECT -- dog bit man)")
print(f"Sentence 2: {' '.join(sentence_2)!r}  (dog is the OBJECT -- man bit dog)")
print()
print(f"max|dog's contextual vector in sentence 1 - dog's contextual vector in sentence 2| "
      f"= {np.max(np.abs(dog_in_1 - dog_in_2)):.2e}")
print("(should be ~0 -- WITHOUT position info, attention gives 'dog' the exact same")
print("representation whether it did the biting or got bitten, because both")
print("sentences contain the identical MULTISET {the, the, dog, man, bit}.)")
print()

# ---------- fix: sinusoidal positional encoding ----------
def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
       PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    Different dimensions oscillate at different frequencies (like a clock
    with hands moving at different speeds) -- combined across all
    dimensions, every position gets a unique, smoothly-varying fingerprint
    that's simply ADDED to the token embedding before attention ever runs.
    """
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle_rates = 1 / (10000 ** ((2 * (i // 2)) / d_model))
    angles = pos * angle_rates
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe


PE = positional_encoding(seq_len=5, d_model=EMBED_DIM)

X1_pos = X1 + PE
X2_pos = X2 + PE

out1_pos = self_attention(X1_pos, Wq, Wk, Wv)
out2_pos = self_attention(X2_pos, Wq, Wk, Wv)

dog_in_1_pos = out1_pos[sentence_1.index("dog")]
dog_in_2_pos = out2_pos[sentence_2.index("dog")]

print("After adding positional encoding to the embeddings before attention:")
print(f"max|dog's contextual vector in sentence 1 - dog's contextual vector in sentence 2| "
      f"= {np.max(np.abs(dog_in_1_pos - dog_in_2_pos)):.4f}")
print("Now clearly nonzero -- 'dog' gets a DIFFERENT representation depending on")
print("whether it's in position 1 (subject slot) or position 4 (object slot),")
print("exactly what a model needs to eventually learn 'who did what to whom.'")
print()

# ---------- visualize the sinusoid pattern ----------
PE_viz = positional_encoding(seq_len=50, d_model=128)
fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(PE_viz.T, aspect="auto", cmap="RdBu", origin="lower")
ax.set_xlabel("Position in sequence")
ax.set_ylabel("Embedding dimension")
ax.set_title("Sinusoidal positional encoding\n(low dims oscillate fast, high dims oscillate slow)")
fig.colorbar(im, ax=ax, label="value")
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "positional_encoding.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved visualization to {out_path}")
