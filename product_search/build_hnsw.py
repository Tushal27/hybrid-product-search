"""
Build an HNSW (approximate nearest neighbour) index from the saved product vectors.

Exact ("flat") search compares the query against all 890k vectors, reading ~1.4 GB of memory per
query -- ~135 ms, and it is memory-bandwidth bound, so extra CPU cores barely help. HNSW instead walks
a graph of "close to each other" links and touches only a few thousand vectors: low milliseconds, at the
cost of occasionally missing a true neighbour. evaluate.py --dense hnsw measures how much that matters.

Build takes several minutes (it inserts 890k vectors one by one into the graph) but is done once.

Usage:  python product_search/build_hnsw.py [M] [efConstruction]
"""

import sys
import time
from pathlib import Path

import faiss
import numpy as np

INDEX_DIR = Path(__file__).parent / "index"
M = int(sys.argv[1]) if len(sys.argv) > 1 else 32
EF_CONSTRUCTION = int(sys.argv[2]) if len(sys.argv) > 2 else 128

vectors = np.load(INDEX_DIR / "vectors.npy").astype(np.float32)
n, dim = vectors.shape
faiss.omp_set_num_threads(10)                      # leave a couple of cores for the rest of the system
index = faiss.IndexHNSWFlat(dim, M, faiss.METRIC_INNER_PRODUCT)
index.hnsw.efConstruction = EF_CONSTRUCTION
start = time.time()
BATCH = 50_000
for i in range(0, n, BATCH):
    index.add(vectors[i:i + BATCH])
    print(f"  added {min(i + BATCH, n):,}/{n:,}  ({(time.time() - start) / 60:.1f} min)", flush=True)
faiss.write_index(index, str(INDEX_DIR / "hnsw.faiss"))
print(f"Saved HNSW (M={M}, efConstruction={EF_CONSTRUCTION}) in {(time.time() - start) / 60:.1f} min -> "
      f"{(INDEX_DIR / 'hnsw.faiss').stat().st_size / 1e9:.2f} GB")
