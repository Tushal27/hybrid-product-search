"""
STEP 1: Why did transformers replace RNNs/LSTMs? Two concrete, measurable
problems with recurrence -- not just "attention is better," but WHY,
demonstrated numerically.

PROBLEM 1: Sequential processing can't parallelize across the sequence.
  An RNN computes hidden_t = f(hidden_{t-1}, input_t). To get hidden_100
  you MUST have already computed hidden_99, which needed hidden_98, etc.
  On a GPU with thousands of cores sitting idle, you can only ever use
  ONE core's worth of work at a time per sequence, because step t+1
  literally cannot start until step t's output exists. Attention computes
  every token's output as a function of ALL tokens simultaneously via
  matrix multiplication -- fully parallel across the sequence dimension.

PROBLEM 2: long-range dependencies have to survive a long chain of
  transformations. Information from token 0 reaching token T in an RNN
  has to pass through T sequential matrix multiplications + nonlinearities.
  If each step shrinks the signal even slightly (very common with typical
  weight initialization), the signal from far back vanishes exponentially.
  In attention, token T can directly attend to token 0 in exactly ONE
  step, regardless of how far apart they are -- path length is O(1), not
  O(distance).

We simulate problem 2 below: a simple RNN-style recurrence with a random
weight matrix (typical init), and measure how much of an initial signal
survives after T steps.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

np.random.seed(0)

HIDDEN_DIM = 64


def simulate_rnn_signal_decay(spectral_scale: float, max_steps: int = 50):
    """hidden_t = tanh(W @ hidden_{t-1}). Start with a unit-norm signal at
    t=0 and track its norm as it propagates -- a stand-in for how much
    'information' from the first token is still recoverable T steps later."""
    W = np.random.randn(HIDDEN_DIM, HIDDEN_DIM) / np.sqrt(HIDDEN_DIM)
    # rescale W so its largest eigenvalue magnitude (spectral radius) is
    # exactly spectral_scale -- this is the knob that decides vanishing
    # (scale < 1) vs exploding (scale > 1) gradients.
    eigvals = np.linalg.eigvals(W)
    W = W / np.max(np.abs(eigvals)) * spectral_scale

    h = np.random.randn(HIDDEN_DIM)
    h = h / np.linalg.norm(h)  # start with unit norm -- a clean "signal"
    norms = [1.0]
    for t in range(max_steps):
        h = np.tanh(W @ h)
        norms.append(np.linalg.norm(h))
    return norms


print("How much of the ORIGINAL signal survives after T recurrent steps,")
print("as a fraction of its starting size, for different weight scalings:")
print()
for label, scale in [("typical init (spectral radius 0.9)", 0.9),
                     ("spectral radius 1.0", 1.0),
                     ("spectral radius 1.1 (exploding)", 1.1)]:
    norms = simulate_rnn_signal_decay(scale)
    print(f"{label}:")
    for t in [1, 5, 10, 20, 50]:
        print(f"    after {t:>2} steps: signal norm = {norms[t]:.6f}")
    print()

print("With spectral radius 0.9 (typical), the signal has shrunk to nearly")
print("nothing by step 20 -- information from token 0 is effectively GONE")
print("by the time token 20 tries to use it, no matter how important it was.")
print("This is the vanishing gradient problem, and it's not a training bug --")
print("it's a structural consequence of forcing information through T sequential")
print("transformations. Getting it exactly right (spectral radius ~1.0, as LSTMs")
print("try to engineer via gating) helps but doesn't eliminate the problem, and")
print("doesn't fix problem 1 (still sequential, still can't parallelize).")
print()

# ---------- contrast: attention's path length is O(1) regardless of distance ----------
print("=" * 70)
print("Attention's answer to both problems: every token's output is a direct,")
print("single-step weighted average over ALL other tokens' value vectors:")
print()
print("    output_i = sum_j  softmax(q_i . k_j / sqrt(d))_j  *  v_j")
print()
print("Token 0 and token 999 are exactly as 'far apart' as token 0 and token 1")
print("from attention's perspective -- there's no chain of intermediate steps")
print("for a signal to decay through. And since every output_i can be computed")
print("independently given all the q/k/v vectors, the whole sequence's outputs")
print("are one batched matrix multiplication -- fully parallel on a GPU.")
print()
print("This tradeoff isn't free: attention pays O(sequence_length^2) compute/memory")
print("(every token compares against every other token) where an RNN pays O(length).")
print("That quadratic cost is exactly why context-window limits and techniques like")
print("KV-caching (week 3 on your roadmap) matter so much in practice.")
 