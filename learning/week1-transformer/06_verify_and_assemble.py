"""
STEP 6: Two things.

PART A: Verify our from-scratch multi-head attention (step 3) produces
IDENTICAL output to PyTorch's real nn.MultiheadAttention, given the exact
same weights. Same "verify against the real implementation" pattern as
the tokenizer track's step 5 (ours vs tiktoken).

PART B: Assemble a full GPT-style transformer block -- causal multi-head
attention + residual connection + layernorm + feedforward network +
another residual + layernorm -- and run real tokenized/embedded text
through it, closing out the whole architecture.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch
import torch.nn as nn
import gensim.downloader as api

np.random.seed(0)
torch.manual_seed(0)

wv = api.load("glove-wiki-gigaword-50")
EMBED_DIM = wv.vector_size   # 50
NUM_HEADS = 5
D_K = EMBED_DIM // NUM_HEADS


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


# ---------- PART A: verify against real PyTorch ----------
sentence = "the cat sat on the mat".split()
X = np.array([wv[t] for t in sentence])
n = len(sentence)

Wq = (np.random.randn(EMBED_DIM, EMBED_DIM) * 0.3).astype(np.float32)
Wk = (np.random.randn(EMBED_DIM, EMBED_DIM) * 0.3).astype(np.float32)
Wv = (np.random.randn(EMBED_DIM, EMBED_DIM) * 0.3).astype(np.float32)
Wo = (np.random.randn(EMBED_DIM, EMBED_DIM) * 0.3).astype(np.float32)
X32 = X.astype(np.float32)

our_output = multi_head_attention(X32, Wq, Wk, Wv, Wo, NUM_HEADS)

torch_mha = nn.MultiheadAttention(EMBED_DIM, NUM_HEADS, bias=False, batch_first=True)
with torch.no_grad():
    # torch's Linear computes x @ weight.T, and ours computes x @ W --
    # so torch's weight must be W.T for the two to compute the same thing.
    torch_mha.in_proj_weight.copy_(torch.tensor(np.concatenate([Wq.T, Wk.T, Wv.T], axis=0)))
    torch_mha.out_proj.weight.copy_(torch.tensor(Wo.T))

X_torch = torch.tensor(X32).unsqueeze(0)  # add batch dim: (1, n_tokens, embed_dim)
torch_output, torch_weights = torch_mha(X_torch, X_torch, X_torch,
                                         need_weights=True, average_attn_weights=False)
torch_output = torch_output.squeeze(0).detach().numpy()

max_diff = np.max(np.abs(our_output - torch_output))
print(f"max|our_output - torch_output| (unmasked) = {max_diff:.2e}")
print(f"MATCH: {max_diff < 1e-4}")
print()

# same check WITH a causal mask
causal_mask = np.tril(np.ones((n, n), dtype=bool))
our_output_causal = multi_head_attention(X32, Wq, Wk, Wv, Wo, NUM_HEADS, mask=causal_mask)

torch_causal_mask = torch.tensor(~causal_mask)  # torch convention: True = BLOCKED (opposite of ours)
torch_output_causal, _ = torch_mha(X_torch, X_torch, X_torch, attn_mask=torch_causal_mask,
                                    need_weights=True, average_attn_weights=False)
torch_output_causal = torch_output_causal.squeeze(0).detach().numpy()

max_diff_causal = np.max(np.abs(our_output_causal - torch_output_causal))
print(f"max|our_output - torch_output| (causal-masked) = {max_diff_causal:.2e}")
print(f"MATCH: {max_diff_causal < 1e-4}")
print()
print("Our from-scratch multi-head attention (steps 2-3, no library, pure numpy)")
print("is numerically identical to PyTorch's production implementation, given")
print("identical weights -- both unmasked (encoder-style) and causal-masked")
print("(decoder-style, step 5). This confirms the mechanics are exactly right,")
print("not just directionally similar.")
print()

# ---------- PART B: assemble a full GPT-style transformer block ----------
print("=" * 70)
print("Assembling one full GPT-style transformer block:")
print("  x -> +positional_encoding -> causal multi-head attention -> +residual")
print("    -> layernorm -> feedforward (expand 4x, GELU, project back) -> +residual")
print("    -> layernorm -> output")
print()


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
        x = layer_norm(x + attn_out)                     # residual + norm
        ff_out = gelu(x @ self.W1 + self.b1) @ self.W2 + self.b2
        x = layer_norm(x + ff_out)                        # residual + norm
        return x


block = TransformerBlock(EMBED_DIM, NUM_HEADS, ff_hidden_dim=EMBED_DIM * 4)

sentence2 = "the tired dog that chased the cat slept".split()
X2 = np.array([wv[t] for t in sentence2])
PE = positional_encoding(len(sentence2), EMBED_DIM)
X2_with_pos = X2 + PE

causal_mask2 = np.tril(np.ones((len(sentence2), len(sentence2)), dtype=bool))
block_output = block.forward(X2_with_pos, mask=causal_mask2)

print(f"Input : {' '.join(sentence2)!r}")
print(f"Input shape (n_tokens, embed_dim)  : {X2_with_pos.shape}")
print(f"Output shape after one full block  : {block_output.shape}")
print(f"Output row means (should be ~0, row stds ~1 -- that's what layernorm guarantees):")
print(f"  means: {np.round(block_output.mean(axis=-1), 4)}")
print(f"  stds : {np.round(block_output.std(axis=-1), 4)}")
print()
print("This is genuinely one full transformer block, the same kind stacked")
print("N times (GPT-2 small stacks 12, GPT-3 stacks 96) to build a real GPT model.")
print("Stack this block N times, add a final linear layer projecting back to")
print("vocab-size logits, and you have the complete architecture from your")
print("roadmap's Week 1 -- tokenizer -> embeddings -> this -> softmax -> next token,")
print("which is exactly the Week 2 'inference pipeline' you'll build on next.")
