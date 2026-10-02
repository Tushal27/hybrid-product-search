"""
Tune the hybrid pipeline's settings on the ESCI TRAINING queries only.
The held-out test queries are never touched here, so the final test score stays honest.

Stage A -- fusion (no reranker). For every training query, run dense and BM25 ONCE and cache
          their top-100 lists; then try many (dense:keyword weight, RRF constant) settings offline
          -- RRF is just arithmetic on the cached lists, so each setting costs milliseconds.
Stage B -- reranker. With the best fusion, rerank each query's top-100 candidates ONCE (the slow
          part), cache the scores, then try different rerank depths and blends offline.

Usage:  python product_search/tune.py [--rerank-queries 600]
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8")
from engine import blend_scores, reciprocal_rank_fusion
from evaluate import ndcg_at_k
from runtime import load_engine

DATA = Path(__file__).parent / "data"
RATIOS = [0.25, 0.5, 1.0, 2.0, 3.0, 5.0]       # dense weight relative to keyword weight (keyword fixed at 1)
RRF_KS = [10, 30, 60, 100]
DEPTHS = [20, 50, 100]
BLENDS = [0.0, 0.25, 0.5, 1.0, 2.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank-queries", type=int, default=600)
    args = ap.parse_args()

    queries = [q for q in json.loads((DATA / "eval_queries.json").read_text()) if q["split"] == "train"]
    print(f"{len(queries)} training queries")
    engine = load_engine()
    pids = engine.catalog["pid"].to_numpy()

    # ---------------- cache dense + BM25 candidate lists
    t = time.time()
    qvecs = engine.embedder.encode_queries([q["query"] for q in queries], batch_size=64)
    cands = []
    for i, q in enumerate(queries):
        d_ids, _ = engine.dense.search(qvecs[i], 100)
        b_ids, _ = engine.bm25.search(q["query"], 100)
        cands.append((d_ids, b_ids))
    print(f"cached candidates in {time.time() - t:.0f}s")

    def mean_ndcg(rankings, qs):
        return float(np.mean([ndcg_at_k(list(pids[r]), q["judgments"]) for r, q in zip(rankings, qs)]))

    dense_only = mean_ndcg([c[0] for c in cands], queries)
    bm25_only = mean_ndcg([c[1] for c in cands], queries)
    print(f"\nbaselines on train: dense {dense_only:.4f} | bm25 {bm25_only:.4f}")

    # ---------------- Stage A: fusion grid
    print("\nStage A: fusion (nDCG@10, train)   rows = dense:keyword weight, cols = RRF k")
    print("            " + "".join(f"{k:>9}" for k in RRF_KS))
    grid, best = {}, (-1, None)
    for ratio in RATIOS:
        row = []
        for k in RRF_KS:
            fused = [[i for i, _ in reciprocal_rank_fusion([d, b], [ratio, 1.0], k=k)] for d, b in cands]
            score = mean_ndcg(fused, queries)
            grid[(ratio, k)] = score
            row.append(score)
            if score > best[0]:
                best = (score, (ratio, k))
        print(f"  {ratio:>5}:1   " + "".join(f"{s:>9.4f}" for s in row))
    best_ratio, best_k = best[1]
    print(f"best fusion: dense:keyword = {best_ratio}:1, RRF k = {best_k}  -> {best[0]:.4f}  (default 1:1, k=60 -> {grid[(1.0, 60)]:.4f})")

    # ---------------- Stage B: reranker
    rng = np.random.default_rng(0)
    subset = [queries[i] for i in rng.choice(len(queries), min(args.rerank_queries, len(queries)), replace=False)]
    idx = {id(q): i for i, q in enumerate(queries)}
    print(f"\nStage B: reranking the top-100 fused candidates of {len(subset)} training queries (the slow part)...", flush=True)
    t = time.time()
    fused_lists, rr_scores, fus_scores = [], [], []
    for n, q in enumerate(subset):
        d, b = cands[idx[id(q)]]
        fused = reciprocal_rank_fusion([d, b], [best_ratio, 1.0], k=best_k)[:100]
        ids = [i for i, _ in fused]
        docs = (engine.catalog["title"].iloc[ids] + ". " + engine.catalog["category"].iloc[ids]).tolist()
        fused_lists.append(ids)
        fus_scores.append(np.array([s for _, s in fused]))
        rr_scores.append(np.asarray(engine.reranker.score(q["query"], docs), dtype=float))
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(subset)}  ({(time.time() - t) / 60:.1f} min)", flush=True)

    no_rerank = mean_ndcg([f for f in fused_lists], subset)
    print(f"\nsame {len(subset)} queries, fusion only (no reranker): {no_rerank:.4f}")
    print("rerank depth x blend (nDCG@10, train subset)   rows = depth, cols = blend alpha")
    print("            " + "".join(f"{a:>9}" for a in BLENDS))
    best_b = (-1, None)
    for depth in DEPTHS:
        row = []
        for alpha in BLENDS:
            rankings = []
            for ids, rr, fs in zip(fused_lists, rr_scores, fus_scores):
                final = blend_scores(rr[:depth], fs[:depth], alpha)
                order = [ids[j] for j in np.argsort(-final)] + ids[depth:]
                rankings.append(order)
            score = mean_ndcg(rankings, subset)
            row.append(score)
            if score > best_b[0]:
                best_b = (score, (depth, alpha))
        print(f"  depth {depth:>3} " + "".join(f"{s:>9.4f}" for s in row))
    depth, alpha = best_b[1]
    print(f"best rerank: depth {depth}, blend {alpha} -> {best_b[0]:.4f}")

    best_settings = dict(dense_weight=best_ratio, bm25_weight=1.0, rrf_k=best_k, rerank_top=depth, rerank_blend=alpha)
    (Path(__file__).parent / "tuned_settings.json").write_text(json.dumps(best_settings, indent=2))
    print("\nSaved", best_settings)


if __name__ == "__main__":
    main()
