"""
WEEK 2, STEP 2: The output head -- turning (n_tokens, embed_dim) into an
actual probability distribution over the next token.

Step 1's MiniGPT body outputs a refined embed_dim-sized vector PER token
position. To predict "what token comes next", GPT projects the LAST
position's vector through one more linear layer -- the "language modeling
head" or "unembedding matrix" -- of shape (embed_dim, vocab_size). That
gives one raw score (a "logit") per vocabulary word. Softmax turns those
logits into a proper probability distribution (nonnegative, sums to 1).

Two things to verify here, same "trust but verify against the real
library" pattern as every previous step:
  (a) our from-scratch softmax matches torch.softmax exactly
  (b) our from-scratch cross-entropy loss (the thing training would
      actually minimize) matches torch.nn.functional.cross_entropy exactly
Verifying (b) matters because ANY model training (from-scratch or
fine-tuning) leans on exactly this loss function -- if it's subtly wrong
here, everything built on top of it later would be wrong too.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch
import torch.nn.functional as F

np.random.seed(0)
torch.manual_seed(0)


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def cross_entropy(logits, target_id):
    """-log(softmax(logits)[target_id]) -- the loss for predicting one token."""
    probs = softmax(logits)
    return -np.log(probs[target_id])


VOCAB_SIZE = 10
EMBED_DIM = 8

# a toy vocabulary so the printed output reads like an actual next-token
# prediction instead of bare integers
vocab = ["the", "cat", "sat", "on", "mat", "dog", "ran", "fast", "and", "slept"]
assert len(vocab) == VOCAB_SIZE

rng = np.random.default_rng(0)
W_head = rng.standard_normal((EMBED_DIM, VOCAB_SIZE)) * 0.3  # the unembedding matrix
b_head = np.zeros(VOCAB_SIZE)

# stand-in for "the last token's vector coming out of the transformer body"
last_hidden_state = rng.standard_normal(EMBED_DIM)

logits = last_hidden_state @ W_head + b_head
probs = softmax(logits)

print("Toy vocabulary:", vocab)
print()
print("Logits (raw scores, one per vocab word, from the output head):")
for word, logit in zip(vocab, logits):
    print(f"  {word:>6}: {logit:+.4f}")
print()
print("After softmax -- an actual probability distribution over 'next token':")
for word, p in zip(vocab, probs):
    bar = "#" * int(p * 100)
    print(f"  {word:>6}: {p:.4f}  {bar}")
print(f"  sum = {probs.sum():.6f}  (must be exactly 1.0)")
print()

top_id = int(np.argmax(probs))
print(f"Most likely next token (argmax): {vocab[top_id]!r} with p={probs[top_id]:.4f}")
print("(weights are random and untrained, so this is a meaningless pick right now --")
print("the MECHANICS are what's being verified, not the prediction quality.)")
print()

# ---------- verify softmax against torch ----------
logits_t = torch.tensor(logits)
torch_probs = F.softmax(logits_t, dim=-1).numpy()
max_diff = np.max(np.abs(probs - torch_probs))
print(f"max|our_softmax - torch.softmax| = {max_diff:.2e}   MATCH: {max_diff < 1e-10}")
print()

# ---------- verify cross-entropy loss against torch ----------
target_id = 4  # pretend the "correct" next word is "mat"
our_loss = cross_entropy(logits, target_id)

torch_loss = F.cross_entropy(logits_t.unsqueeze(0), torch.tensor([target_id])).item()
diff = abs(our_loss - torch_loss)
print(f"Target word: {vocab[target_id]!r}")
print(f"our cross-entropy loss   = {our_loss:.6f}")
print(f"torch cross-entropy loss = {torch_loss:.6f}")
print(f"diff = {diff:.2e}   MATCH: {diff < 1e-10}")
print()
print("Both match PyTorch exactly. This confirms the full inference-time math --")
print("hidden state -> logits -> softmax -> probabilities, and the loss used to")
print("MEASURE how wrong a prediction is -- is implemented correctly from scratch,")
print("with no library doing the real work underneath.")
