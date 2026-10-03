"""
Measure the speed/quality trade-offs behind the serving optimisations, on the held-out test queries.

Part 1  HNSW vs exact dense search -- per-query latency, how many of the true top-10 / top-100 neighbours
        HNSW finds (overlap), and the nDCG@10 of dense-only ranking with each.
Part 2  Reranker variants -- fp32 vs int8-quantised, reranking the top 15 / 25 / 50 fused candidates:
        milliseconds per query, nDCG@10 (with the tuned fusion + blend), and how much of the fp32-depth-50
        top-10 each variant reproduces.

Usage:  python product_search/bench_stages.py [--n 497]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8")
from engine import DenseIndex, INDEX_DIR, Reranker, blend_scores, reciprocal_rank_fusion
from evaluate import ndcg_at_k
from runtime import load_engine

DATA = Path(__file__).parent / "data"
TUNED = json.loads((Path(__file__).parent / "tuned_settings.json").read_text())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--skip-exact", action="store_true", help="skip the exact-search ground truth (Part 1 numbers already known)")
    ap.add_argument("--variants", default="fp32:50,fp32:25,int8:50,int8:25,int8:15", help="name:depth list; the first one is the reference")
    args = ap.parse_args()
    queries = [q for q in json.loads((DATA / "eval_queries.json").read_text(encoding="utf-8")) if q["split"] == "test"]
    if args.n:
        queries = queries[:args.n]
    print(f"{len(queries)} test queries")

    engine = load_engine(dense_kind="hnsw", rerank=True)          # HNSW dense + fp32 reranker
    hnsw = engine.dense
    pids = engine.catalog["pid"].to_numpy()
    qvecs = engine.embedder.encode_queries([q["query"] for q in queries], batch_size=64)

    # ------------------------------------------------------------ Part 1: HNSW vs exact
    print("\nPart 1: dense search, HNSW vs exact (flat)")
    if args.skip_exact:
        res = {"hnsw": [hnsw.search(qv, 100)[0] for qv in qvecs]}
    else:
        # Exact ground truth WITHOUT holding a second full copy of the vectors in RAM (the machine is tight on memory):
        # score all queries against the fp16 vectors in 100k-row chunks and keep each query's running top-100.
        vec16 = np.load(INDEX_DIR / "vectors.npy", mmap_mode="r")
        Q = qvecs.astype(np.float32)
        best_s = np.full((len(Q), 100), -np.inf, dtype=np.float32)
        best_i = np.zeros((len(Q), 100), dtype=np.int64)
        for start in range(0, len(vec16), 100_000):
            chunk = np.asarray(vec16[start:start + 100_000], dtype=np.float32)
            sc = Q @ chunk.T
            s_all = np.concatenate([best_s, sc], axis=1)
            i_all = np.concatenate([best_i, np.arange(start, start + len(chunk))[None, :].repeat(len(Q), 0)], axis=1)
            top = np.argpartition(-s_all, 99, axis=1)[:, :100]
            best_s = np.take_along_axis(s_all, top, 1)
            best_i = np.take_along_axis(i_all, top, 1)
        order = np.argsort(-best_s, axis=1)
        exact = [row[o] for row, o in zip(best_i, order)]
        del vec16

        hnsw_ids, hnsw_ms = [], []
        for qv in qvecs:
            t = time.perf_counter()
            ids, _ = hnsw.search(qv, 100)
            hnsw_ms.append((time.perf_counter() - t) * 1000)
            hnsw_ids.append(ids)
        res = {"flat": exact, "hnsw": hnsw_ids}
        for name in ("flat", "hnsw"):
            nd = np.mean([ndcg_at_k(list(pids[r]), q["judgments"]) for r, q in zip(res[name], queries)])
            extra = f"search p50 {np.percentile(hnsw_ms, 50):5.1f} ms  p95 {np.percentile(hnsw_ms, 95):5.1f} ms" if name == "hnsw" else "(exact; ~135 ms per query measured earlier)"
            print(f"  {name:<5} dense nDCG@10 {nd:.4f}   {extra}")
        o10 = np.mean([len(set(a[:10]) & set(b[:10])) / 10 for a, b in zip(res["flat"], res["hnsw"])])
        o100 = np.mean([len(set(a) & set(b)) / 100 for a, b in zip(res["flat"], res["hnsw"])])
        print(f"  HNSW finds {o10:.1%} of the exact top-10 and {o100:.1%} of the exact top-100", flush=True)


    # ------------------------------------------------------------ Part 2: reranker variants
    print("\nPart 2: reranker variants (fusion = tuned settings; candidates come from HNSW + BM25)")
    cands = []
    for q, qv, d_ids in zip(queries, qvecs, res["hnsw"]):
        b_ids, _ = engine.bm25.search(q["query"], 100)
        cands.append(reciprocal_rank_fusion([d_ids, b_ids], [TUNED["dense_weight"], TUNED["bm25_weight"]], k=TUNED["rrf_k"])[:100])

    base = [list(pids[[i for i, _ in c]]) for c in cands]
    print(f"  no reranker (fusion only)     nDCG@10 {np.mean([ndcg_at_k(b, q['judgments']) for b, q in zip(base, queries)]):.4f}")

    rerankers = {"fp32": engine.reranker, "int8": Reranker(device="cpu", quantize=True)}
    reference = None
    for name, depth in [(v.split(":")[0], int(v.split(":")[1])) for v in args.variants.split(",")]:
        rr_model = rerankers[name]
        ms, tops, nd = [], [], []
        for q, c in zip(queries, cands):
            ids = [i for i, _ in c]
            fus = np.array([s for _, s in c])
            docs = (engine.catalog["title"].iloc[ids[:depth]] + ". " + engine.catalog["category"].iloc[ids[:depth]]).tolist()
            t = time.perf_counter()
            sc = np.asarray(rr_model.score(q["query"], docs), dtype=float)
            ms.append((time.perf_counter() - t) * 1000)
            final = blend_scores(sc, fus[:depth], TUNED["rerank_blend"])
            order = [ids[j] for j in np.argsort(-final)] + ids[depth:]
            ranked = list(pids[order])
            tops.append(set(ranked[:10]))
            nd.append(ndcg_at_k(ranked, q["judgments"]))
        if reference is None:
            reference = tops
        agree = np.mean([len(a & b) / 10 for a, b in zip(reference, tops)])
        print(f"  {name} depth {depth:>2}: rerank p50 {np.percentile(ms, 50):6.0f} ms  p95 {np.percentile(ms, 95):6.0f} ms   "
              f"nDCG@10 {np.mean(nd):.4f}   same top-10 as fp32/50: {agree:.1%}", flush=True)


if __name__ == "__main__":
    main()
