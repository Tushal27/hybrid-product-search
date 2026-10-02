"""
Build the evaluation set: real shopper queries + human relevance labels, restricted to our toy catalog.

Source: ESCI (Amazon Shopping Queries). Each (query, product) pair was labelled by people as
Exact / Substitute / Complement / Irrelevant. We keep:
  * US queries only (our catalog is English / US Amazon),
  * only labelled products that exist in our toys catalog,
  * only queries that are genuinely about toys -- at least half of the products labelled for the
    query are toys (otherwise "bathroom fan" would sneak in because one fan listing is a toy),
  * only queries with at least one Exact match we can actually find.

Gains for scoring follow the ESCI benchmark: Exact=1.0, Substitute=0.1, Complement=0.01, Irrelevant=0.

HONEST CAVEAT: ESCI judged only the products Amazon's own search showed for each query. A relevant
toy that was never shown has no label and counts as irrelevant. This under-reports every method's
absolute scores equally -- trust the *comparison between methods* more than the absolute numbers.

Usage:  python product_search/build_eval.py
"""

import glob
import json
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

sys.stdout.reconfigure(encoding="utf-8")
DATA = Path(__file__).parent / "data"
GAIN = {"Exact": 1.0, "Substitute": 0.1, "Complement": 0.01, "Irrelevant": 0.0}
COLS = ["query", "query_id", "product_id", "product_locale", "esci_label"]

catalog_pids = set(pd.read_parquet(DATA / "catalog.parquet", columns=["pid"])["pid"])
queries = []
for split in ("train", "test"):
    frames = []
    for f in sorted(glob.glob(str(DATA / "esci_raw" / "data" / f"{split}-*.parquet"))):
        df = pq.read_table(f, columns=COLS).to_pandas()
        frames.append(df[df["product_locale"] == "us"])
    if not frames:
        continue
    df = pd.concat(frames, ignore_index=True)
    df["is_toy"] = df["product_id"].isin(catalog_pids)
    toy_frac = df.groupby("query_id")["is_toy"].mean()
    keep_ids = toy_frac[toy_frac >= 0.5].index
    toys = df[df["is_toy"] & df["query_id"].isin(keep_ids)]
    for qid, g in toys.groupby("query_id"):
        if (g["esci_label"] == "Exact").any():
            queries.append({
                "split": split, "query_id": int(qid), "query": g["query"].iloc[0],
                "judgments": {p: GAIN[l] for p, l in zip(g["product_id"], g["esci_label"])},
            })
    print(f"{split}: {df['query_id'].nunique():,} US queries -> {sum(q['split'] == split for q in queries):,} toy queries kept")

out = DATA / "eval_queries.json"
out.write_text(json.dumps(queries))
n_judg = [len(q["judgments"]) for q in queries]
print(f"Saved {len(queries)} queries -> {out}   (avg {sum(n_judg) / max(len(n_judg), 1):.1f} labelled toys per query)")
for q in queries[:6]:
    print(f"  {q['query']!r}: {len(q['judgments'])} labelled, {sum(v == 1.0 for v in q['judgments'].values())} exact")
