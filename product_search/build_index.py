"""
Build the search indexes from product_search/data/catalog.parquet (run catalog.py first).

  * dense vectors: every product's text -> a 384-number vector. This is the slow part
    (~30+ minutes for ~890k products), so it runs on the GPU in chunks of 50k saved to
    disk as it goes -- if it is interrupted, re-running resumes where it stopped.
  * BM25 keyword index: fast (a couple of minutes), CPU only.

Usage:
    python product_search/build_index.py              # everything
    python product_search/build_index.py --bm25-only  # skip the embedding step
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8")
from catalog import load_catalog
from embedder import Embedder
from engine import BM25Index, INDEX_DIR, SearchEngine

CHUNK = 50_000
CHUNK_DIR = Path(__file__).parent / "data" / "emb_chunks"


def build_vectors(texts):
    CHUNK_DIR.mkdir(parents=True, exist_ok=True)
    embedder = Embedder(device="cuda")
    n_chunks = (len(texts) + CHUNK - 1) // CHUNK
    start = time.time()
    for c in range(n_chunks):
        path = CHUNK_DIR / f"chunk_{c:04d}.npy"
        if path.exists():
            continue
        vecs = embedder.encode_documents(texts[c * CHUNK:(c + 1) * CHUNK], batch_size=256)
        np.save(path, vecs.astype(np.float16))                       # fp16 on disk: half the size, no retrieval loss
        done = c + 1
        print(f"  embedded chunk {done}/{n_chunks}  ({(time.time() - start) / 60:.1f} min elapsed)", flush=True)
    vectors = np.concatenate([np.load(CHUNK_DIR / f"chunk_{c:04d}.npy") for c in range(n_chunks)])
    assert len(vectors) == len(texts)
    np.save(INDEX_DIR / "vectors.npy", vectors)
    print(f"Saved vectors {vectors.shape} -> {INDEX_DIR / 'vectors.npy'}")


def main():
    INDEX_DIR.mkdir(exist_ok=True)
    catalog = load_catalog()
    texts = catalog["text"].tolist()
    print(f"{len(texts):,} products")
    if "--bm25-only" not in sys.argv:
        print("Embedding on GPU...")
        build_vectors(texts)
    t = time.time()
    print("Building BM25 keyword index...")
    BM25Index(texts).save(INDEX_DIR / "bm25")
    print(f"BM25 done in {time.time() - t:.0f}s")
    SearchEngine.save_meta(INDEX_DIR, {"n_products": len(texts), "embedding_model": "BAAI/bge-small-en-v1.5", "dim": 384})


if __name__ == "__main__":
    main()
