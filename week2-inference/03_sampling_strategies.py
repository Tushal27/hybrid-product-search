"""
WEEK 2, STEP 3: How does a GPT actually pick the NEXT token from the
probability distribution step 2 produced? There's more than one strategy,
and this is exactly where "same model, different behavior" comes from --
e.g. why ChatGPT's "temperature" setting changes how creative vs
repetitive it sounds, using the SAME trained weights.

GREEDY: always pick argmax. Deterministic, often repetitive/boring, and
  can get stuck ("the the the the...").
TEMPERATURE: divide logits by T before softmax. T<1 sharpens the
  distribution (more confident/greedy-like), T>1 flattens it (more
  random/diverse). T=1 is a no-op -- exactly the original distribution.
TOP-K: zero out every probability except the k highest, renormalize, then
  sample. Caps how "weird" a pick can be, regardless of how flat the
  distribution is.
TOP-P (nucleus): zero out everything except the smallest set of highest
  -probability tokens whose probabilities sum to >= p, renormalize, then
  sample. Adapts the cutoff per-step -- keeps few tokens when the model is
  confident, more tokens when it's uncertain (unlike top-k's fixed count).

All of this is applied to a FIXED example logit vector (not run through
the real model) so the effect of each strategy is easy to see in
isolation, with a fixed random seed so the sampled picks are reproducible.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

rng = np.random.default_rng(0)

vocab = ["the", "cat", "sat", "on", "mat", "dog", "ran", "fast", "and", "slept"]
# a deliberately peaked-but-not-certain distribution, like a partially-trained
# model that's fairly confident "the" comes next but not 100% sure
logits = np.array([3.0, 0.5, 2.2, 0.2, 1.8, -0.5, 0.8, 1.0, -1.0, -0.8])


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def print_dist(label, probs):
    print(f"{label}:")
    order = np.argsort(-probs)
    for i in order:
        if probs[i] < 1e-4:
            continue
        bar = "#" * int(probs[i] * 100)
        print(f"    {vocab[i]:>6}: {probs[i]:.4f}  {bar}")
    print()


# ---------- baseline ----------
base_probs = softmax(logits)
print_dist("Baseline distribution (T=1, no filtering)", base_probs)


# ---------- greedy ----------
def greedy(logits):
    return int(np.argmax(logits))


greedy_pick = greedy(logits)
print(f"GREEDY pick: {vocab[greedy_pick]!r}  (always the same token, every time,")
print("no randomness involved at all)")
print()


# ---------- temperature ----------
def apply_temperature(logits, T):
    return softmax(logits / T)


for T in [0.5, 1.0, 2.0]:
    probs_T = apply_temperature(logits, T)
    print_dist(f"TEMPERATURE T={T}", probs_T)
print("Lower T concentrates probability onto the already-favored token(s) (T->0")
print("approaches greedy); higher T flattens the distribution toward uniform")
print("(more of the 'weird' tokens become plausible picks).")
print()


# ---------- top-k ----------
def top_k_filter(logits, k):
    top_k_idx = np.argsort(-logits)[:k]
    filtered = np.full_like(logits, -np.inf)
    filtered[top_k_idx] = logits[top_k_idx]
    return softmax(filtered)


for k in [1, 3, 5]:
    probs_k = top_k_filter(logits, k)
    nonzero = int(np.sum(probs_k > 1e-4))
    print_dist(f"TOP-K k={k}  ({nonzero} tokens have nonzero probability)", probs_k)


# ---------- top-p (nucleus) ----------
def top_p_filter(logits, p):
    probs = softmax(logits)
    order = np.argsort(-probs)
    sorted_probs = probs[order]
    cumulative = np.cumsum(sorted_probs)
    # keep the smallest prefix whose cumulative probability >= p
    cutoff = np.searchsorted(cumulative, p) + 1
    keep_idx = order[:cutoff]
    filtered = np.full_like(logits, -np.inf)
    filtered[keep_idx] = logits[keep_idx]
    return softmax(filtered)


for p in [0.5, 0.9]:
    probs_p = top_p_filter(logits, p)
    nonzero = int(np.sum(probs_p > 1e-4))
    print_dist(f"TOP-P p={p}  ({nonzero} tokens kept -- this count ADAPTS to how "
               f"peaked/flat the distribution is, unlike top-k's fixed count)", probs_p)


# ---------- actually sampling (not just showing the distribution) ----------
def sample(probs, rng):
    return int(rng.choice(len(probs), p=probs))


print("=" * 70)
print("Drawing 10 samples from each strategy (fixed seed, so reproducible):")
strategies = {
    "greedy (T~0)": lambda: greedy_pick,
    "temperature T=1.0": lambda: sample(base_probs, rng),
    "temperature T=2.0": lambda: sample(apply_temperature(logits, 2.0), rng),
    "top-k k=3": lambda: sample(top_k_filter(logits, 3), rng),
    "top-p p=0.9": lambda: sample(top_p_filter(logits, 0.9), rng),
}
for name, fn in strategies.items():
    picks = [vocab[fn()] for _ in range(10)]
    print(f"  {name:<20}: {picks}")
print()
print("Greedy never varies. Temperature=1.0 and top-p/top-k draw from the SAME")
print("underlying model but visibly vary the output -- this is the actual mechanism")
print("behind every 'temperature' / 'creativity' slider you've seen in an LLM UI.")
