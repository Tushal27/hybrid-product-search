"""
Extract one thumbnail URL per product (pid -> url) for the web UI. The search itself doesn't use images;
this only makes the live page easier to judge by eye.

Usage:  python product_search/build_images.py      -> product_search/data/images.parquet
"""

import glob
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

DATA = Path(__file__).parent / "data"


def first_url(img):
    if img is None:
        return None
    for key in ("large", "thumb", "hi_res"):
        urls = img.get(key) if hasattr(img, "get") else None
        if urls is not None and len(urls):
            for u in urls:
                if u:
                    return u
    return None


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    frames = []
    for path in sorted(glob.glob(str(DATA / "toys_raw" / "raw_meta_Toys_and_Games" / "*.parquet"))):
        df = pq.read_table(path, columns=["parent_asin", "images"]).to_pandas()
        frames.append(pd.DataFrame({"pid": df["parent_asin"], "image": df["images"].map(first_url)}))
        print(f"  {Path(path).name}", flush=True)
    out = pd.concat(frames).drop_duplicates("pid").dropna()
    out.to_parquet(DATA / "images.parquet")
    print(f"{len(out):,} products have an image -> {DATA / 'images.parquet'}")
