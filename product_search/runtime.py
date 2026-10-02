"""Load a ready-to-query SearchEngine from the saved indexes (CPU only -- this is the serving path)."""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from catalog import load_catalog
from embedder import Embedder
from engine import BM25Index, DenseIndex, INDEX_DIR, Reranker, SearchEngine


def load_engine(dense_kind="flat", rerank=True, device="cpu", verbose=True):
    t = time.time()
    catalog = load_catalog()
    vectors = np.load(INDEX_DIR / "vectors.npy").astype(np.float32)
    assert len(vectors) == len(catalog), "vectors and catalog are out of sync -- rebuild the index"
    dense = DenseIndex(vectors, kind=dense_kind)
    bm25 = BM25Index.load(INDEX_DIR / "bm25")
    embedder = Embedder(device=device)
    reranker = Reranker(device=device) if rerank else None
    if verbose:
        print(f"Engine ready: {len(catalog):,} products, dense={dense_kind}, rerank={'on' if rerank else 'off'} "
              f"({time.time() - t:.0f}s to load)")
    return SearchEngine(catalog, dense, bm25, embedder, reranker)
