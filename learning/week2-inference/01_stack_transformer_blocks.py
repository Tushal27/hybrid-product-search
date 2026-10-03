"""
WEEK 2, STEP 1: Stack transformer blocks into an actual (untrained) GPT.

Week 1 ended with ONE transformer block: attention + residual + norm +
feedforward + residual + norm. A real GPT is just that block repeated N
times back to back, each one refining the same (n_tokens, embed_dim)
representation a little further -- GPT-2 small stacks 12, GPT-2 XL stacks
48, GPT-3 stacks 96. Nothing architecturally new happens by stacking; the
only new ingredient here is a learned TOKEN EMBEDDING TABLE (previously we
used pretrained GloVe vectors directly as input -- a real GPT starts from
its OWN embedding table, randomly initialized, and learns it during
training) plus the same sinusoidal positional encoding added on top.

This script has no training yet (that's a later week) -- weights are
random, so the numbers coming out mean nothing semantically. The point is
purely mechanical: prove the shapes flow correctly through N stacked
blocks and that each block is actually doing something (changing the
representation), not a no-op.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

np.random.seed(0)


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
        weights = softmax(scores)
        head_outputs.append(weights @ V_h)
    concatenated = np.concatenate(head_outputs, axis=-1)
    return concatenated @ Wo


def layer_norm(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle_rates = 1 / (10000 ** ((2 * (i // 2)) / d_model))
    angles = pos * angle_rates
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe


class TransformerBlock:
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, seed=0):
        rng = np.random.default_rng(seed)
        scale = 0.3
        self.Wq = rng.standard_normal((embed_dim, embed_dim)) * scale
        self.Wk = rng.standard_normal((embed_dim, embed_dim)) * scale
        self.Wv = rng.standard_normal((embed_dim, embed_dim)) * scale
        self.Wo = rng.standard_normal((embed_dim, embed_dim)) * scale
        self.W1 = rng.standard_normal((embed_dim, ff_hidden_dim)) * scale
        self.b1 = np.zeros(ff_hidden_dim)
        self.W2 = rng.standard_normal((ff_hidden_dim, embed_dim)) * scale
        self.b2 = np.zeros(embed_dim)
        self.num_heads = num_heads

    def forward(self, x, mask=None):
        attn_out = multi_head_attention(x, self.Wq, self.Wk, self.Wv, self.Wo,
                                         self.num_heads, mask=mask)
        x = layer_norm(x + attn_out)
        ff_out = gelu(x @ self.W1 + self.b1) @ self.W2 + self.b2
        x = layer_norm(x + ff_out)
        return x


class MiniGPT:
    """Token embedding -> +positional encoding -> N transformer blocks -> final norm."""

    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_hidden_dim,
                 max_seq_len, seed=0):
        rng = np.random.default_rng(seed)
        # a real GPT's token embedding table: (vocab_size, embed_dim), learned
        # from scratch during training. Small init like GPT-2's actual 0.02 std.
        self.token_embed = rng.standard_normal((vocab_size, embed_dim)) * 0.02
        self.pos_encoding = positional_encoding(max_seq_len, embed_dim)
        self.blocks = [TransformerBlock(embed_dim, num_heads, ff_hidden_dim, seed=seed + i)
                        for i in range(num_layers)]
        self.num_layers = num_layers

    def forward(self, token_ids, verbose=False):
        n_tokens = len(token_ids)
        x = self.token_embed[token_ids] + self.pos_encoding[:n_tokens]
        mask = np.tril(np.ones((n_tokens, n_tokens), dtype=bool))
        if verbose:
            print(f"  after embed + pos_encoding : shape={x.shape}, "
                  f"mean={x.mean():.4f}, std={x.std():.4f}")
        for layer_idx, block in enumerate(self.blocks):
            x_prev = x
            x = block.forward(x, mask=mask)
            if verbose:
                change = np.abs(x - x_prev).mean()
                print(f"  after block {layer_idx + 1:>2}/{self.num_layers}          : "
                      f"shape={x.shape}, mean={x.mean():.4f}, std={x.std():.4f}, "
                      f"avg change from prev layer={change:.4f}")
        x = layer_norm(x)  # final layernorm, same as GPT-2's ln_f
        return x


VOCAB_SIZE = 50
EMBED_DIM = 32
NUM_HEADS = 4
NUM_LAYERS = 6
FF_HIDDEN_DIM = EMBED_DIM * 4
MAX_SEQ_LEN = 16

model = MiniGPT(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_HIDDEN_DIM, MAX_SEQ_LEN)

token_ids = np.array([3, 17, 8, 41, 2, 9])  # stand-in for real token ids -- no real tokenizer wired up yet

print(f"MiniGPT: vocab_size={VOCAB_SIZE}, embed_dim={EMBED_DIM}, num_heads={NUM_HEADS}, "
      f"num_layers={NUM_LAYERS}, ff_hidden_dim={FF_HIDDEN_DIM}")
print(f"Input token ids: {token_ids.tolist()}  (length {len(token_ids)})")
print()
print("Forward pass through the stack:")
output = model.forward(token_ids, verbose=True)
print()
print(f"Final output shape: {output.shape}  (still (n_tokens, embed_dim) -- stacking")
print("blocks never changes the shape, only refines the representation. Note each")
print("layer's 'avg change from prev layer' is nonzero and roughly stable: every")
print("block is doing real work, not collapsing to a no-op or exploding.")
print()
print(f"Row means ~0, row stds ~1 (guaranteed by the final layernorm):")
print(f"  means: {np.round(output.mean(axis=-1), 4)}")
print(f"  stds : {np.round(output.std(axis=-1), 4)}")
print()
print("This IS the 'body' of a GPT model. What's missing to turn it into an actual")
print("language model: (1) a real tokenizer mapping text -> token_ids (week 1 built")
print("this already), (2) an output head projecting embed_dim -> vocab_size logits")
print("so we get a probability distribution over the NEXT token (step 2), and")
print("(3) real weights instead of random init -- either trained from scratch, or")
print("(the far more common path in practice) loaded from an already-pretrained")
print("open model, which is what most of this pipeline is actually built for.")
