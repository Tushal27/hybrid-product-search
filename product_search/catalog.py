"""
Turn the raw Amazon "Toys & Games" metadata shards into one clean table.

Each product becomes a row with display fields (title, brand, price, rating)
and ONE search text. Real catalogs are messy: many products have no bullet
points or description at all, so the text is assembled from whatever exists --
title first (it carries most of the signal), then brand, category, features.

Usage:  python product_search/catalog.py        -> product_search/data/catalog.parquet
"""

import glob
import re
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

DATA = Path(__file__).parent / "data"
CATALOG_PATH = DATA / "catalog.parquet"
COLUMNS = ["parent_asin", "title", "store", "price", "average_rating", "rating_number",
           "features", "description", "categories", "main_category"]
MAX_SEARCH_CHARS = 700          # embedding model reads ~128 tokens anyway; keep BM25 text bounded too


def _clean(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def _first(items, n, max_chars):
    """First n non-empty strings of a list-like, each trimmed."""
    out = []
    for x in (items if items is not None else []):
        x = _clean(x)
        if x:
            out.append(x[:max_chars])
        if len(out) == n:
            break
    return out


def _price(p):
    m = re.search(r"\d+(?:,\d{3})*(?:\.\d+)?", str(p))
    return float(m.group().replace(",", "")) if m else None


def build_search_text(row):
    parts = [_clean(row["title"])]
    if row["store"] and str(row["store"]) not in ("None", "nan"):
        parts.append(f"Brand: {_clean(row['store'])}")
    cats = _first(row["categories"], 4, 60)
    if cats:
        parts.append("Category: " + " > ".join(cats))
    parts.extend(_first(row["features"], 3, 200))
    if not cats and not _first(row["features"], 1, 5):          # nothing else known -> fall back to the description
        parts.extend(_first(row["description"], 1, 300))
    return ". ".join(p for p in parts if p)[:MAX_SEARCH_CHARS]


def build_catalog():
    shards = sorted(glob.glob(str(DATA / "toys_raw" / "raw_meta_Toys_and_Games" / "*.parquet")))
    if not shards:
        sys.exit("No catalog shards found -- run product_search/download_data.py first.")
    frames = []
    for path in shards:
        df = pq.read_table(path, columns=COLUMNS).to_pandas()
        df = df[df["title"].notna() & (df["title"].str.strip() != "")]
        frames.append(pd.DataFrame({
            "pid": df["parent_asin"],
            "title": df["title"].map(_clean),
            "brand": df["store"].where(df["store"].notna(), None),
            "price": df["price"].map(_price),
            "rating": df["average_rating"],
            "n_ratings": df["rating_number"],
            "category": df["categories"].map(lambda c: " > ".join(_first(c, 3, 40))),
            "text": df.apply(build_search_text, axis=1),
        }))
        print(f"  {Path(path).name}: {len(frames[-1]):,} products", flush=True)
    catalog = pd.concat(frames, ignore_index=True).drop_duplicates("pid").reset_index(drop=True)
    catalog.to_parquet(CATALOG_PATH)
    print(f"Saved {len(catalog):,} products -> {CATALOG_PATH}")
    return catalog


def load_catalog():
    return pd.read_parquet(CATALOG_PATH)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    cat = build_catalog()
    print(cat.sample(3, random_state=0).T.to_string(max_colwidth=110))
    print("\nsearch-text length (chars): ", cat["text"].str.len().describe()[["mean", "50%", "max"]].round(0).to_dict())
