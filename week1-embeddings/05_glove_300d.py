"""
STEP 5: Same experiments as steps 3-4, but with glove-wiki-gigaword-300
instead of the 50-dimensional version. 300 dimensions means the model had
6x more "room" to encode distinctions between words during training --
worth seeing directly whether that shows up as tighter, more separated
clustering, not just taking it on faith.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import gensim.downloader as api

print("Loading glove-wiki-gigaword-300 (downloads ~376MB on first run, cached after)...")
wv = api.load("glove-wiki-gigaword-300")
print(f"Loaded. Vocab size: {len(wv.key_to_index):,}, dimensions: {wv.vector_size}")
print()


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# ---------- 1. rerun the related-vs-unrelated gap test from step 3 ----------
related_words = [("cat", "dog"), ("cat", "kitten"), ("king", "queen"),
                  ("happy", "glad"), ("car", "vehicle"), ("doctor", "nurse")]
unrelated_words = [("cat", "banana"), ("king", "spreadsheet"), ("happy", "cement"),
                    ("car", "philosophy"), ("doctor", "guitar"), ("dog", "algebra")]


def avg_similarity(pairs):
    sims = [cosine_similarity(wv[a], wv[b]) for a, b in pairs]
    return np.mean(sims), np.std(sims), sims


related_mean, related_std, _ = avg_similarity(related_words)
unrelated_mean, unrelated_std, _ = avg_similarity(unrelated_words)
print(f"300D: related mean={related_mean:+.3f} (std={related_std:.3f})   "
      f"unrelated mean={unrelated_mean:+.3f} (std={unrelated_std:.3f})   "
      f"gap={related_mean - unrelated_mean:+.3f}")
print("(compare to 50D's gap of +0.730 from step 3)")
print()

# ---------- 2. king - man + woman analogy ----------
print("king - man + woman ~= ?")
for word, score in wv.most_similar(positive=["king", "woman"], negative=["man"], topn=5):
    print(f"  {word:>10}  (similarity {score:.3f})")
print()

# ---------- 3. the 'mouse' ambiguity check from step 4 ----------
print("Nearest neighbors of 'mouse' (animal vs computer-peripheral ambiguity):")
for neighbor, score in wv.most_similar("mouse", topn=8):
    print(f"  {neighbor:>12}  {score:.3f}")
print()

# ---------- 4. PCA plot, same word groups as step 4, for direct comparison ----------
groups = {
    "animals":  ["cat", "dog", "horse", "lion", "tiger", "elephant", "mouse"],
    "countries": ["france", "germany", "italy", "spain", "japan", "china", "india"],
    "emotions": ["happy", "sad", "angry", "joyful", "furious", "depressed", "excited"],
    "tech":     ["computer", "software", "internet", "algorithm", "database", "server"],
}
colors = {"animals": "tab:blue", "countries": "tab:orange", "emotions": "tab:green", "tech": "tab:red"}

words, vectors, group_labels = [], [], []
for group, word_list in groups.items():
    for w in word_list:
        words.append(w)
        vectors.append(wv[w])
        group_labels.append(group)

vectors = np.array(vectors)
pca = PCA(n_components=2)
coords_2d = pca.fit_transform(vectors)
print(f"Original dimensionality: {vectors.shape[1]}")
print(f"Variance captured by 2 principal components: {pca.explained_variance_ratio_.sum():.1%}")
print("(compare to 50D's 43.0% from step 4 -- fewer of 300 dims can be squeezed into just 2)")
print()

fig, ax = plt.subplots(figsize=(10, 8))
for group in groups:
    mask = [g == group for g in group_labels]
    xs = coords_2d[mask, 0]
    ys = coords_2d[mask, 1]
    ax.scatter(xs, ys, c=colors[group], label=group, s=80, alpha=0.8)
    for x, y, w in zip(xs, ys, [w for w, g in zip(words, group_labels) if g == group]):
        ax.annotate(w, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)

ax.set_title("Real GloVe embeddings (300D -> 2D via PCA)\n"
              "same word groups as the 50D plot -- compare directly")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.legend()
ax.grid(alpha=0.3)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embedding_clusters_300d.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved plot to {out_path}")
