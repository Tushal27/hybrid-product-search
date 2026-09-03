"""
STEP 2: How do you measure "similar meaning" between two vectors? Cosine
similarity. Built from scratch here (no library shortcut) so the mechanics
are fully visible.

cosine_similarity(a, b) = (a . b) / (|a| * |b|)

  a . b   = dot product = sum(a_i * b_i for each dimension i)
          = large and positive when a and b point in similar directions,
            near zero when they're perpendicular (unrelated),
            negative when they point in opposite directions.
  |a|,|b| = vector magnitudes (Euclidean length) -- dividing by these
            normalizes for vector SIZE, so we're comparing DIRECTION only,
            not magnitude. This matters because nothing in embeddings
            guarantees vectors will have equal length, but we usually care
            about "do these point the same way," not "which is longer."

Result ranges from -1 (opposite) to 1 (identical direction), 0 = unrelated.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

np.random.seed(0)


def dot(a: np.ndarray, b: np.ndarray) -> float:
    total = 0.0
    for x, y in zip(a, b):        # written as an explicit loop on purpose --
        total += x * y            # this IS what np.dot does internally
    return total


def magnitude(a: np.ndarray) -> float:
    return dot(a, a) ** 0.5       # sqrt(sum of squares) = Euclidean length


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return dot(a, b) / (magnitude(a) * magnitude(b))


# sanity check against numpy's own (much faster, vectorized) implementation
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 1.0, 0.0])
ours = cosine_similarity(a, b)
numpys = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
print(f"our cosine_similarity: {ours:.6f}")
print(f"numpy equivalent     : {numpys:.6f}")
print(f"match: {abs(ours - numpys) < 1e-9}")
print()

# intuition check with hand-built vectors before touching real embeddings
identical = cosine_similarity(np.array([1, 2, 3]), np.array([1, 2, 3]))
same_direction = cosine_similarity(np.array([1, 2, 3]), np.array([2, 4, 6]))     # scaled copy
perpendicular = cosine_similarity(np.array([1, 0]), np.array([0, 1]))
opposite = cosine_similarity(np.array([1, 2, 3]), np.array([-1, -2, -3]))
print(f"identical vectors        -> similarity = {identical:.3f}  (expect 1.0)")
print(f"same direction, 2x scale -> similarity = {same_direction:.3f}  (expect 1.0 -- magnitude is ignored!)")
print(f"perpendicular vectors    -> similarity = {perpendicular:.3f}  (expect 0.0)")
print(f"opposite direction       -> similarity = {opposite:.3f}  (expect -1.0)")
print()

# ---------- now the actual point: RANDOM embeddings are meaningless ----------
# Use the REAL tokenizer to get ids -- no hardcoded/guessed token ids -- and
# verify each word is actually a single token before using it.
import os, importlib.util
_tok_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "week1-tokenizer")
_spec = importlib.util.spec_from_file_location("verify", os.path.join(_tok_dir, "05_verify_gpt4.py"))
verify = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify)
tok = verify.OurGPT4Tokenizer()


def single_token_id(word: str) -> int:
    ids = tok.encode(" " + word)
    assert len(ids) == 1, f"{word!r} isn't a single token: {ids}"
    return ids[0]


VOCAB_SIZE = 100_277
EMBED_DIM = 8
embedding_matrix = np.random.randn(VOCAB_SIZE, EMBED_DIM) * 0.02

# One cherry-picked triple proves nothing -- at only 8 dimensions, random
# cosine similarity has a standard deviation of roughly 1/sqrt(8) ~ 0.35
# from chance ALONE. So instead: many "should be related" pairs and many
# "should be unrelated" pairs, compare the AVERAGE similarity of each group.
# If random init carried real signal, related pairs would average
# noticeably higher -- if it's noise, the two group averages should be
# statistically indistinguishable.
related_words = [("cat", "dog"), ("cat", "kitten"), ("king", "queen"),
                  ("happy", "glad"), ("car", "vehicle"), ("doctor", "nurse")]
unrelated_words = [("cat", "banana"), ("king", "spreadsheet"), ("happy", "cement"),
                    ("car", "philosophy"), ("doctor", "guitar"), ("dog", "algebra")]


def avg_similarity(pairs):
    sims = []
    for a, b in pairs:
        ida, idb = single_token_id(a), single_token_id(b)
        sims.append(cosine_similarity(embedding_matrix[ida], embedding_matrix[idb]))
    return np.mean(sims), np.std(sims), sims


related_mean, related_std, related_sims = avg_similarity(related_words)
unrelated_mean, unrelated_std, unrelated_sims = avg_similarity(unrelated_words)

print("Cosine similarity between RANDOM (untrained) embedding rows:")
print(f"  'related' word pairs   : mean={related_mean:+.3f}  std={related_std:.3f}  {[round(float(s),2) for s in related_sims]}")
print(f"  'unrelated' word pairs : mean={unrelated_mean:+.3f}  std={unrelated_std:.3f}  {[round(float(s),2) for s in unrelated_sims]}")
print()
print(f"Difference in means: {related_mean - unrelated_mean:+.3f} -- tiny compared to the")
print(f"~{related_std:.2f} spread WITHIN each group. These two groups are statistically")
print("indistinguishable: random init cannot tell 'cat/dog' from 'cat/banana' on average,")
print("even though one lucky/unlucky individual pair might look suggestive by chance")
print("(that's exactly the trap of eyeballing a single example instead of a group).")
print("Meaning comes entirely from training. Next: load REAL trained vectors and watch")
print("this same test produce a large, consistent gap between the two groups.")
