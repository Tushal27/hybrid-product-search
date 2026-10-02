"""
Command-line front end.

    python product_search/cli.py search "remote control car" --k 5 --max-price 30
    python product_search/cli.py search "lego star wars" --mode bm25
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.stdout.reconfigure(encoding="utf-8")
from runtime import load_engine


def main():
    ap = argparse.ArgumentParser(description="Toy product search")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--k", type=int, default=10)
    s.add_argument("--mode", default="hybrid_rerank", choices=["dense", "bm25", "hybrid", "hybrid_rerank"])
    s.add_argument("--min-price", type=float)
    s.add_argument("--max-price", type=float)
    s.add_argument("--min-rating", type=float)
    s.add_argument("--brand")
    args = ap.parse_args()

    engine = load_engine(rerank=args.mode == "hybrid_rerank")
    resp = engine.search(args.query, k=args.k, mode=args.mode, min_price=args.min_price,
                         max_price=args.max_price, min_rating=args.min_rating, brand=args.brand)
    for r in resp.results:
        price = f"${r.price:.2f}" if r.price is not None else "  n/a "
        rating = f"{r.rating:.1f}*" if r.rating is not None else " n/a"
        print(f"{r.rank:>3}. {price:>8}  {rating}  {r.title[:90]}  [{r.brand or '-'}]")
    print(f"\n{len(resp.results)} results, {resp.n_candidates} candidates considered | timings (ms): {resp.timings_ms}")


if __name__ == "__main__":
    main()
