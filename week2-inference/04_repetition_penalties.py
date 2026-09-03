"""
WEEK 2, STEP 4: Frequency penalty and presence penalty -- the other half
of "decoding strategies", alongside temperature/top-k/top-p (step 3).

Temperature/top-k/top-p all shape the distribution using ONLY the current
step's logits -- they have no memory of what's already been generated.
That's exactly why a model can still loop into "the the the the..." even
with sampling turned on, if "the" keeps winning the random draw. Frequency
and presence penalties fix this by looking at GENERATION HISTORY and
pushing already-used tokens down before sampling.

FREQUENCY PENALTY: subtract penalty * count(token) from that token's
  logit, where count(token) is how many times it has ALREADY appeared in
  the generated text so far. Scales with repetition -- a token used 5
  times gets punished more than one used once. This is what actually
  breaks "the the the the" loops and stutter/repetition.

PRESENCE PENALTY: subtract a FLAT penalty from a token's logit if it has
  appeared AT LEAST ONCE before, regardless of how many times. Doesn't
  care about count, only about "have we seen this at all" -- this is what
  pushes a model toward introducing NEW topics/words instead of circling
  back to ones already used, even just once.

(These are literally the `frequency_penalty` and `presence_penalty`
parameters in the OpenAI/Anthropic-style chat completion APIs -- this is
what's actually happening under the hood when you set them.)
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
from collections import Counter

rng = np.random.default_rng(0)

vocab = ["the", "cat", "sat", "on", "mat", "dog", "ran", "fast", "and", "slept"]
# same base logits for every token position -- an intentionally repetitive
# model that always wants to say "the" most, to make the penalties' effect
# obvious and isolate them from any other randomness
base_logits = np.array([3.0, 0.5, 2.2, 0.2, 1.8, -0.5, 0.8, 1.0, -1.0, -0.8])


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def apply_penalties(logits, generated_ids, freq_penalty=0.0, presence_penalty=0.0):
    counts = Counter(generated_ids)
    penalized = logits.copy()
    for token_id, count in counts.items():
        penalized[token_id] -= freq_penalty * count        # scales with how many times it's shown up
        penalized[token_id] -= presence_penalty * 1         # flat, only cares "has it shown up at all"
    return penalized


def greedy_generate(base_logits, n_steps, freq_penalty=0.0, presence_penalty=0.0):
    generated = []
    for _ in range(n_steps):
        logits = apply_penalties(base_logits, generated, freq_penalty, presence_penalty)
        pick = int(np.argmax(logits))
        generated.append(pick)
    return generated


N_STEPS = 12

print("Base logits always favor 'the' the most -- an exaggerated stand-in for a")
print("model stuck in a repetition loop:")
for word, logit in zip(vocab, base_logits):
    print(f"  {word:>6}: {logit:+.2f}")
print()

print("=" * 70)
print("NO penalty (greedy, same logits every step -> classic repetition loop):")
seq = greedy_generate(base_logits, N_STEPS)
print("  " + " ".join(vocab[i] for i in seq))
print()

print("FREQUENCY PENALTY = 0.8 (punishment grows with each repeat use):")
seq = greedy_generate(base_logits, N_STEPS, freq_penalty=0.8)
print("  " + " ".join(vocab[i] for i in seq))
print()

print("PRESENCE PENALTY = 1.5 (flat punishment the moment a word is used ONCE):")
seq = greedy_generate(base_logits, N_STEPS, presence_penalty=1.5)
print("  " + " ".join(vocab[i] for i in seq))
print()

print("BOTH combined (frequency=0.5, presence=1.0):")
seq = greedy_generate(base_logits, N_STEPS, freq_penalty=0.5, presence_penalty=1.0)
print("  " + " ".join(vocab[i] for i in seq))
print()

# ---------- show the actual logit math for one concrete step ----------
print("=" * 70)
print("What's actually happening to the logits, one step in:")
generated_so_far = [0, 0, 2]  # "the the sat"
print(f"Generated so far: {[vocab[i] for i in generated_so_far]}")
print(f"  -> counts: {dict((vocab[k], v) for k, v in Counter(generated_so_far).items())}")
print()

for label, fp, pp in [("no penalty", 0.0, 0.0), ("frequency=0.8", 0.8, 0.0), ("presence=1.5", 0.0, 1.5)]:
    penalized = apply_penalties(base_logits, generated_so_far, freq_penalty=fp, presence_penalty=pp)
    print(f"{label}:")
    for word, orig, pen in zip(vocab, base_logits, penalized):
        marker = "  <-- was penalized" if orig != pen else ""
        print(f"    {word:>6}: {orig:+.2f} -> {pen:+.2f}{marker}")
    print()

print("Frequency penalty hits 'the' (used twice) HARDER than 'sat' (used once) --")
print("the punishment scales with count. Presence penalty hits 'the' and 'sat' by")
print("the exact SAME flat amount -- it only asks 'seen before, yes/no', not 'how")
print("many times'. Both combine cleanly with temperature/top-k/top-p from step 3:")
print("penalties adjust the logits FIRST based on history, then temperature/top-k/")
print("top-p shape the resulting distribution for sampling -- that's the full")
print("decoding-strategy picture from the roadmap's week 2 box.")
