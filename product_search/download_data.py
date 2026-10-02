"""
Download the two public datasets the search module is built and tested on.

  * Amazon Reviews 2023, "Toys & Games" product metadata (~890k products) -- the catalog we search.
  * ESCI (Amazon Shopping Queries): real shopper queries with human relevance labels
    (Exact / Substitute / Complement / Irrelevant) -- the ground truth for measuring search quality.

Usage:  python product_search/download_data.py
"""

from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

print("Toys & Games catalog...")
for i in range(5):
    name = f"raw_meta_Toys_and_Games/full-{i:05d}-of-00005.parquet"
    hf_hub_download("McAuley-Lab/Amazon-Reviews-2023", name, repo_type="dataset", local_dir=DATA / "toys_raw")
    print("  got", name, flush=True)

print("ESCI relevance judgments...")
for name in list_repo_files("tasksource/esci", repo_type="dataset"):
    if name.endswith(".parquet"):
        hf_hub_download("tasksource/esci", name, repo_type="dataset", local_dir=DATA / "esci_raw")
        print("  got", name, flush=True)
print("All downloaded.")
