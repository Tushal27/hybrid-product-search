"""
Measure search quality on real labelled shopper queries (built by build_eval.py).

Metrics, per query then averaged:
  nDCG@10    -- the headline number. Rewards putting Exact matches first, gives partial credit to
                Substitutes, and discounts results further down the list. 1.0 = perfect ordering.
  MRR@10     -- 1 / rank of the first Exact match. "How far down must the shopper scroll?"
  Recall@100 -- share of the Exact matches that appear anywhere in the top 100. For the candidate
                stages this is the ceiling: a reranker cannot rescue a product retrieval never found.
Latency is wall-clock per query on this machine's CPU, including query embedding and (if used) reranking.

Usage:
    python product_search/evaluate.py                      # test split, all modes
    python product_search/evaluate.py --n 300 --modes dense,hybrid
    python product_search/evaluate.py --split train --dense hnsw
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8")

DATA = Path(__file__).parent / "data"


def ndcg_at_k(ranked_pids, judgments, k=10):
    dcg = sum(judgments.get(p, 0.0) / math.log2(i + 2) for i, p in enumerate(ranked_pids[:k]))
    ideal = sorted(judgments.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mrr_at_k(ranked_pids, judgments, k=10):
    for i, p in enumerate(ranked_pids[:k]):
        if judgments.get(p, 0.0) == 1.0:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked_pids, judgments, k=100):
    exact = {p for p, g in judgments.items() if g == 1.0}
    return len(exact & set(ranked_pids[:k])) / len(exact) if exact else 0.0


def evaluate(engine, queries, mode, **search_kwargs):
    ndcg, mrr, rec, lat = [], [], [], []
    for q in queries:
        t = time.perf_counter()
        resp = engine.search(q["query"], k=100, mode=mode, **search_kwargs)
        lat.append((time.perf_counter() - t) * 1000)
        pids = [r.pid for r in resp.results]
        ndcg.append(ndcg_at_k(pids, q["judgments"]))
        mrr.append(mrr_at_k(pids, q["judgments"]))
        rec.append(recall_at_k(pids, q["judgments"]))
    return {"mode": mode, "n": len(queries), "ndcg@10": float(np.mean(ndcg)), "mrr@10": float(np.mean(mrr)),
            "recall@100": float(np.mean(rec)), "p50_ms": float(np.percentile(lat, 50)), "p95_ms": float(np.percentile(lat, 95)),
            "_ndcg_per_query": ndcg}


def bootstrap_ci(values, n_boot=2000, seed=0):
    """95% confidence interval of the mean, by resampling queries with replacement."""
    v = np.asarray(values)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_diff(a, b, n_boot=2000, seed=0):
    """b minus a, per query, bootstrapped. Pairing matters: both methods saw the SAME queries, so
    query difficulty cancels out and small real gaps become visible."""
    d = np.asarray(b) - np.asarray(a)
    lo, hi = bootstrap_ci(d, n_boot, seed)
    return float(d.mean()), lo, hi


def print_table(rows):
    print(f"\n{'config':<22}{'nDCG@10':>9}{'  95% CI':>16}{'MRR@10':>9}{'Recall@100':>12}{'p50 ms':>9}{'p95 ms':>9}")
    for r in rows:
        lo, hi = bootstrap_ci(r["_ndcg_per_query"])
        print(f"{r['mode']:<22}{r['ndcg@10']:>9.3f}{f'[{lo:.3f},{hi:.3f}]':>16}{r['mrr@10']:>9.3f}"
              f"{r['recall@100']:>12.3f}{r['p50_ms']:>9.0f}{r['p95_ms']:>9.0f}")


def print_paired(rows):
    by = {r["mode"]: r for r in rows}
    pairs = [("dense", "hybrid_rerank"), ("dense", "hybrid"), ("hybrid", "hybrid_rerank"),
             ("hybrid_rerank", "hybrid_rerank_tuned"), ("dense", "hybrid_rerank_tuned"), ("hybrid", "hybrid_tuned")]
    print("\nPaired nDCG@10 difference (second minus first), 95% CI. If the interval excludes 0, the gap is real, not noise:")
    for a, b in pairs:
        if a in by and b in by:
            m, lo, hi = paired_diff(by[a]["_ndcg_per_query"], by[b]["_ndcg_per_query"])
            verdict = "real" if lo > 0 or hi < 0 else "NOT distinguishable from noise"
            print(f"  {b:<22} - {a:<14} {m:+.4f}  [{lo:+.4f}, {hi:+.4f}]  {verdict}")


BASELINE = dict(dense_weight=1.0, bm25_weight=1.0, rrf_k=60, rerank_top=50, rerank_blend=0.0)   # the original untuned settings


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="test")
    ap.add_argument("--n", type=int, default=0, help="use only the first N queries (0 = all)")
    ap.add_argument("--modes", default="dense,bm25,hybrid,hybrid_rerank", help="comma list; *_tuned variants are added automatically")
    ap.add_argument("--dense", default="flat", choices=["flat", "hnsw"])
    args = ap.parse_args()

    queries = [q for q in json.loads((DATA / "eval_queries.json").read_text()) if q["split"] == args.split]
    if args.n:
        queries = queries[:args.n]
    print(f"{len(queries)} {args.split} queries")
    from runtime import load_engine      # imported here, not at the top: the metric functions need no ML libraries (keeps tests/CI light)
    engine = load_engine(dense_kind=args.dense)
    engine.search("warmup", k=5, mode="hybrid_rerank")        # first call pays one-off setup costs

    tuned_path = Path(__file__).parent / "tuned_settings.json"
    tuned = json.loads(tuned_path.read_text()) if tuned_path.exists() else None
    configs = [(m, m, BASELINE) for m in args.modes.split(",")]
    if tuned:
        configs += [(f"{m}_tuned", m, tuned) for m in args.modes.split(",") if m in ("hybrid", "hybrid_rerank")]
        print("tuned settings:", tuned)

    rows = []
    for name, mode, kwargs in configs:
        print(f"evaluating {name}...", flush=True)
        row = evaluate(engine, queries, mode, **kwargs)
        row["mode"] = name
        rows.append(row)
    print_table(rows)
    print_paired(rows)
    out = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    (Path(__file__).parent / "results.json").write_text(json.dumps({"split": args.split, "dense": args.dense, "tuned": tuned, "rows": out}, indent=2))


if __name__ == "__main__":
    main()
