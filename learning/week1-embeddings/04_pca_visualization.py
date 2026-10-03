"""
STEP 4: The picture from your roadmap image -- "similar meaning, closer in
vector space" -- made real instead of illustrative.

GloVe vectors are 50-dimensional; humans can't look at 50 numbers and see
a cluster. PCA (Principal Component Analysis) finds the 2 directions in
that 50D space that capture the MOST variance, and projects every vector
onto just those 2 directions so we can actually plot it. This necessarily
throws away information (50D -> 2D), but if the clustering by category
still shows up clearly, that's strong evidence the structure is real and
dominant in the data, not some artifact of a specific pair of dimensions.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display needed, just save to file
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import gensim.downloader as api

wv = api.load("glove-wiki-gigaword-50")

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
print("(i.e. we're keeping this much of the original information after squashing 50D->2D)")
print()

fig, ax = plt.subplots(figsize=(10, 8))
for group in groups:
    mask = [g == group for g in group_labels]
    xs = coords_2d[mask, 0]
    ys = coords_2d[mask, 1]
    ax.scatter(xs, ys, c=colors[group], label=group, s=80, alpha=0.8)
    for x, y, w in zip(xs, ys, [w for w, g in zip(words, group_labels) if g == group]):
        ax.annotate(w, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)

ax.set_title("Real GloVe embeddings, 50D -> 2D via PCA\n"
              "(same idea as your roadmap image, but with actual trained vectors)")
ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.legend()
ax.grid(alpha=0.3)

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "embedding_clusters.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved plot to {out_path}")
