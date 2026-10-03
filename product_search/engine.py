"""
The search engine: three retrieval stages that each fix the last one's blind spot.

  1. DENSE   (embedding vectors)  -- understands MEANING: "toy for a 3 year old who loves trucks"
                                     finds a "dump truck for toddlers" with no shared words.
                                     Weak on exact tokens: model numbers, "lego 75192", rare names.
  2. BM25    (keyword statistics) -- the opposite: great at exact terms, blind to synonyms.
  3. FUSION  (reciprocal rank)    -- merges the two ranked lists; items both like float to the top.
  4. RERANK  (cross-encoder)      -- reads query AND product together for the best ~50 candidates and
                                     re-scores them. Slow per item but only run on a few, so it is
                                     affordable, and it is the biggest single quality gain.

Filters (price / rating / brand) are applied INSIDE the dense and BM25 searches
(not after), so asking for "top 10 under $20" really returns 10 results.
"""

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import bm25s
import faiss
import numpy as np
import pandas as pd

try:
    import Stemmer
    _stemmer = Stemmer.Stemmer("english")
except ImportError:   # works without it, just slightly weaker keyword matching
    _stemmer = None

INDEX_DIR = Path(__file__).parent / "index"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60            # standard reciprocal-rank-fusion constant

# Tuned on the ESCI *training* queries (see tune.py) -- never on the held-out test queries.
DEFAULTS = dict(dense_weight=2.0, bm25_weight=1.0, rrf_k=10, rerank_top=50, rerank_blend=0.5)


# ------------------------------------------------------------------ dense (vectors)
class DenseIndex:
    """Nearest-neighbour search over product vectors. 'flat' = exact (brute force); 'hnsw' = approximate graph, faster."""

    def __init__(self, vectors, kind="flat", hnsw_m=32, ef_construction=200, ef_search=128):
        self.n, self.dim = vectors.shape
        self.kind = kind
        if kind == "hnsw":
            self.index = faiss.IndexHNSWFlat(self.dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
            self.index.hnsw.efConstruction = ef_construction
            self.index.hnsw.efSearch = ef_search
        else:
            self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(np.ascontiguousarray(vectors, dtype=np.float32))

    @classmethod
    def from_saved(cls, path, kind="hnsw", ef_search=128):
        obj = cls.__new__(cls)
        obj.index = faiss.read_index(str(path))
        obj.kind, obj.n, obj.dim = kind, obj.index.ntotal, obj.index.d
        if kind == "hnsw":
            obj.index.hnsw.efSearch = ef_search
        return obj

    def search(self, qvec, k, mask=None):
        params = None
        if mask is not None:   # restrict the search to allowed products, inside FAISS
            bitmap = np.packbits(mask, bitorder="little")
            sel = faiss.IDSelectorBitmap(self.n, faiss.swig_ptr(bitmap))
            params = faiss.SearchParametersHNSW() if self.kind == "hnsw" else faiss.SearchParameters()
            params.sel = sel
            if self.kind == "hnsw":
                params.efSearch = max(self.index.hnsw.efSearch, k * 4)
        scores, ids = self.index.search(qvec.reshape(1, -1).astype(np.float32), k, params=params)
        ok = ids[0] >= 0
        return ids[0][ok], scores[0][ok]


# ------------------------------------------------------------------ keyword (BM25)
class BM25Index:
    def __init__(self, texts=None):
        self.retriever = bm25s.BM25()
        if texts is not None:
            tokens = bm25s.tokenize(list(texts), stopwords="en", stemmer=_stemmer, show_progress=False)
            self.retriever.index(tokens, show_progress=False)

    def search(self, query, k, mask=None):
        q = bm25s.tokenize([query], stopwords="en", stemmer=_stemmer, show_progress=False)
        if not len(q.vocab):
            return np.array([], dtype=int), np.array([], dtype=float)
        weight_mask = None if mask is None else mask.astype(np.float32)
        ids, scores = self.retriever.retrieve(q, k=k, show_progress=False, weight_mask=weight_mask)
        keep = scores[0] > 0
        return ids[0][keep], scores[0][keep]

    def save(self, path):
        self.retriever.save(str(path))

    @classmethod
    def load(cls, path):
        obj = cls()
        obj.retriever = bm25s.BM25.load(str(path), load_corpus=False)
        return obj


# ------------------------------------------------------------------ fusion + rerank
def reciprocal_rank_fusion(rankings, weights=None, k=RRF_K):
    """Merge several ranked id lists. Score = sum of weight / (k + rank). Only ranks matter, so BM25 and
    cosine scores (which live on totally different scales) never need to be normalised against each other."""
    weights = weights or [1.0] * len(rankings)
    fused = {}
    for ranking, w in zip(rankings, weights):
        for rank, doc in enumerate(ranking):
            fused[int(doc)] = fused.get(int(doc), 0.0) + w / (k + rank + 1)
    return sorted(fused.items(), key=lambda kv: -kv[1])


def blend_scores(rerank_scores, fusion_scores, alpha):
    """Final order for the reranked candidates: standardised reranker score + alpha * standardised fusion score.
    alpha=0 trusts the reranker completely; larger alpha keeps more of the first-stage order."""
    def z(x):
        sd = x.std()
        return (x - x.mean()) / sd if sd > 0 else np.zeros_like(x)
    return z(rerank_scores) + alpha * z(fusion_scores)


class Reranker:
    def __init__(self, model_name=RERANK_MODEL, device="cpu", max_length=192, quantize=False):
        from sentence_transformers import CrossEncoder
        self.model = CrossEncoder(model_name, device=device, max_length=max_length)
        if quantize:   # store Linear weights as int8, matmuls run in int8: faster on CPU, tiny accuracy cost (measured in bench_stages.py)
            import torch    # in place on the underlying BERT: sentence-transformers 6 chains modules, so swapping `.model` breaks it
            torch.quantization.quantize_dynamic(self.model.transformers_model, {torch.nn.Linear}, dtype=torch.qint8, inplace=True)

    def score(self, query, docs, batch_size=32):
        return self.model.predict([(query, d) for d in docs], batch_size=batch_size, show_progress_bar=False)


# ------------------------------------------------------------------ the engine
@dataclass
class SearchResult:
    pid: str
    title: str
    brand: str | None
    price: float | None
    rating: float | None
    category: str
    score: float
    rank: int


@dataclass
class SearchResponse:
    results: list
    timings_ms: dict = field(default_factory=dict)
    n_candidates: int = 0


class SearchEngine:
    MODES = ("dense", "bm25", "hybrid", "hybrid_rerank")

    def __init__(self, catalog, dense, bm25, embedder, reranker=None):
        self.catalog, self.dense, self.bm25, self.embedder, self.reranker = catalog, dense, bm25, embedder, reranker
        # HF fast tokenizers raise "Already borrowed" if two threads use one at once, so each model gets its own
        # lock. FAISS and BM25 searches are thread-safe and run unlocked -- while request A is inside the
        # reranker, request B can already be doing its dense + keyword search.
        self._embed_lock, self._rerank_lock = threading.Lock(), threading.Lock()
        self._brand_lower = catalog["brand"].fillna("").str.lower().to_numpy()
        self._price = catalog["price"].to_numpy(dtype=float)       # NaN where unknown
        self._rating = catalog["rating"].to_numpy(dtype=float)

    # ---- filters -> boolean mask over the whole catalog (None = no filtering)
    def _mask(self, min_price, max_price, min_rating, brand):
        if min_price is None and max_price is None and min_rating is None and not brand:
            return None
        mask = np.ones(len(self.catalog), dtype=bool)
        if min_price is not None:
            mask &= self._price >= min_price           # unknown price (NaN) never passes a price filter
        if max_price is not None:
            mask &= self._price <= max_price
        if min_rating is not None:
            mask &= self._rating >= min_rating
        if brand:
            mask &= self._brand_lower == brand.lower()
        return mask

    def search(self, query, k=10, mode="hybrid_rerank", candidates=100, rerank_top=None,
               min_price=None, max_price=None, min_rating=None, brand=None,
               dense_weight=None, bm25_weight=None, rrf_k=None, rerank_blend=None):
        rerank_top = DEFAULTS["rerank_top"] if rerank_top is None else rerank_top
        dense_weight = DEFAULTS["dense_weight"] if dense_weight is None else dense_weight
        bm25_weight = DEFAULTS["bm25_weight"] if bm25_weight is None else bm25_weight
        rrf_k = DEFAULTS["rrf_k"] if rrf_k is None else rrf_k
        rerank_blend = DEFAULTS["rerank_blend"] if rerank_blend is None else rerank_blend
        assert mode in self.MODES, f"mode must be one of {self.MODES}"
        t = {}
        mask = self._mask(min_price, max_price, min_rating, brand)
        pool = max(candidates, k)
        dense_ids = bm25_ids = np.array([], dtype=int)
        scores = {}

        if mode in ("dense", "hybrid", "hybrid_rerank"):
            t0 = time.perf_counter()
            with self._embed_lock:
                qvec = self.embedder.encode_queries([query])[0]
            t["embed_query"] = (time.perf_counter() - t0) * 1000
            t0 = time.perf_counter()
            dense_ids, dense_scores = self.dense.search(qvec, pool, mask)
            t["dense_search"] = (time.perf_counter() - t0) * 1000
            scores.update({int(i): float(s) for i, s in zip(dense_ids, dense_scores)})
        if mode in ("bm25", "hybrid", "hybrid_rerank"):
            t0 = time.perf_counter()
            bm25_ids, bm25_scores = self.bm25.search(query, pool, mask)
            t["bm25_search"] = (time.perf_counter() - t0) * 1000
            if mode == "bm25":
                scores.update({int(i): float(s) for i, s in zip(bm25_ids, bm25_scores)})

        if mode == "dense":
            ranked = [(int(i), scores[int(i)]) for i in dense_ids]
        elif mode == "bm25":
            ranked = [(int(i), scores[int(i)]) for i in bm25_ids]
        else:
            t0 = time.perf_counter()
            ranked = reciprocal_rank_fusion([dense_ids, bm25_ids], [dense_weight, bm25_weight], k=rrf_k)
            t["fusion"] = (time.perf_counter() - t0) * 1000

        n_candidates = len(ranked)
        if mode == "hybrid_rerank" and self.reranker is not None and ranked:
            t0 = time.perf_counter()
            top = [i for i, _ in ranked[:rerank_top]]
            docs = (self.catalog["title"].iloc[top] + ". " + self.catalog["category"].iloc[top]).tolist()
            with self._rerank_lock:
                rr = np.asarray(self.reranker.score(query, docs), dtype=float)
            final = blend_scores(rr, np.array([s for _, s in ranked[:rerank_top]]), rerank_blend)
            ranked = sorted(zip(top, map(float, final)), key=lambda kv: -kv[1]) + ranked[rerank_top:]
            t["rerank"] = (time.perf_counter() - t0) * 1000

        rows = self.catalog.iloc[[i for i, _ in ranked[:k]]]
        results = [
            SearchResult(pid=r.pid, title=r.title, brand=r.brand, price=None if pd.isna(r.price) else float(r.price),
                         rating=None if pd.isna(r.rating) else float(r.rating), category=r.category,
                         score=float(s), rank=n + 1)
            for n, (r, (_, s)) in enumerate(zip(rows.itertuples(), ranked[:k]))
        ]
        t["total"] = sum(t.values())
        return SearchResponse(results=results, timings_ms={k_: round(v, 1) for k_, v in t.items()}, n_candidates=n_candidates)

    # ---- build / save / load
    @staticmethod
    def save_meta(index_dir, info):
        (Path(index_dir) / "meta.json").write_text(json.dumps(info, indent=2))
