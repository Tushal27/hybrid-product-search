"""
STEP 3: Load REAL trained word embeddings (GloVe, trained on Wikipedia +
Gigaword news text, 50 dimensions) and rerun the EXACT same experiment as
step 2 -- related-word-pairs vs unrelated-word-pairs cosine similarity.

Step 2's random-init embeddings gave: related mean=-0.041, unrelated
mean=-0.039 -- no real gap, just noise. If embeddings genuinely capture
meaning, trained vectors should show a large, consistent gap instead.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import gensim.downloader as api

print("Loading glove-wiki-gigaword-50 (downloads ~66MB on first run, cached after)...")
wv = api.load("glove-wiki-gigaword-50")
print(f"Loaded. Vocab size: {len(wv.key_to_index):,}, dimensions: {wv.vector_size}")
print()


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


related_words = [("cat", "dog"), ("cat", "kitten"), ("king", "queen"),
                  ("happy", "glad"), ("car", "vehicle"), ("doctor", "nurse")]
unrelated_words = [("cat", "banana"), ("king", "spreadsheet"), ("happy", "cement"),
                    ("car", "philosophy"), ("doctor", "guitar"), ("dog", "algebra")]


def avg_similarity(pairs):
    sims = [cosine_similarity(wv[a], wv[b]) for a, b in pairs]
    return np.mean(sims), np.std(sims), sims


related_mean, related_std, related_sims = avg_similarity(related_words)
unrelated_mean, unrelated_std, unrelated_sims = avg_similarity(unrelated_words)

print("Cosine similarity between REAL (GloVe-trained) embedding rows:")
print(f"  'related' word pairs   : mean={related_mean:+.3f}  std={related_std:.3f}  "
      f"{[round(float(s), 2) for s in related_sims]}")
print(f"  'unrelated' word pairs : mean={unrelated_mean:+.3f}  std={unrelated_std:.3f}  "
      f"{[round(float(s), 2) for s in unrelated_sims]}")
print()
print(f"Gap between group means: {related_mean - unrelated_mean:+.3f}")
print(f"(compare to step 2's random-init gap of just -0.003)")
print()

# the classic analogy: king - man + woman ~= queen
print("Classic vector-arithmetic analogy: king - man + woman ~= ?")
result = wv.most_similar(positive=["king", "woman"], negative=["man"], topn=5)
for word, score in result:
    print(f"  {word:>10}  (similarity {score:.3f})")
print()

# nearest neighbors of a word -- shows real clustering by meaning
for word in ["cat", "computer"]:
    print(f"Nearest neighbors of {word!r}:")
    for neighbor, score in wv.most_similar(word, topn=6):
        print(f"  {neighbor:>12}  {score:.3f}")
    print()

print("This is the entire difference training makes: same lookup-table")
print("mechanism as step 1, same cosine-similarity math as step 2, but the")
print("NUMBERS actually mean something now because they were fit to predict")
print("real co-occurrence patterns across billions of words of real text.")
