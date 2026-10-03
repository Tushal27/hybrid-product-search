"""
Load test for the search API.

OPEN-LOOP: requests arrive on a schedule (Poisson, like real independent shoppers) no matter how
slowly the server answers. A closed-loop tester ("send the next one when the last returns") quietly
slows itself down when the server struggles and hides the very overload we want to see.

Queries are real shopper queries from the ESCI set. Real traffic is skewed -- a few queries are very
common -- so --zipf controls how often popular queries repeat (0 = all distinct, ~1 = realistic).

For each arrival rate it reports: throughput actually achieved, error rate (HTTP errors + timeouts),
and latency p50/p95/p99. Latency is measured from the moment the request was SCHEDULED, so time spent
waiting in a queue counts -- that is what a user feels.

Usage:
    python product_search/loadtest.py --rates 1,2,4,8 --duration 30
    python product_search/loadtest.py --rates 20,50,100 --duration 20 --zipf 1.0 --mode hybrid_rerank
Run it from a second terminal while api.py is running. It shares this machine's CPU with the server,
so for the cleanest numbers keep everything else idle.
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import httpx
import numpy as np

sys.stdout.reconfigure(encoding="utf-8")
DATA = Path(__file__).parent / "data"


def load_queries():
    qs = json.loads((DATA / "eval_queries.json").read_text(encoding="utf-8"))
    return [q["query"] for q in qs]


async def run_rate(client, url, queries, rate, duration, zipf, mode, timeout, rng):
    n = max(1, int(rate * duration))
    arrivals = np.cumsum(rng.exponential(1.0 / rate, size=n))             # Poisson arrival times (seconds)
    if zipf > 0:
        weights = 1.0 / np.arange(1, len(queries) + 1) ** zipf
        picks = rng.choice(len(queries), size=n, p=weights / weights.sum())
    else:
        picks = rng.integers(0, len(queries), size=n)
    results = []

    async def one(i, t_sched):
        status, ok = 0, False
        try:
            r = await client.get(url + "/search", params={"q": queries[picks[i]], "k": 10, "mode": mode}, timeout=timeout)
            status, ok = r.status_code, r.status_code == 200
        except httpx.TimeoutException:
            status = -1
        except httpx.HTTPError:
            status = -2
        results.append((time.perf_counter() - t_sched, ok, status))

    t0 = time.perf_counter()
    tasks = []
    for i, a in enumerate(arrivals):
        delay = t0 + a - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(one(i, t0 + a)))
    await asyncio.gather(*tasks)
    wall = time.perf_counter() - t0

    lat = np.array([r[0] for r in results if r[1]]) * 1000
    codes = {}
    for _, ok, st in results:
        if not ok:
            codes[st] = codes.get(st, 0) + 1
    return {
        "rate": rate, "sent": n, "ok": len(lat), "errors": n - len(lat), "error_codes": codes,
        "throughput": len(lat) / wall,
        "p50": float(np.percentile(lat, 50)) if len(lat) else None,
        "p95": float(np.percentile(lat, 95)) if len(lat) else None,
        "p99": float(np.percentile(lat, 99)) if len(lat) else None,
        "max": float(lat.max()) if len(lat) else None,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--rates", default="1,2,4,8", help="comma list of requests/second to test, in order")
    ap.add_argument("--duration", type=float, default=30)
    ap.add_argument("--zipf", type=float, default=1.0)
    ap.add_argument("--mode", default="hybrid_rerank")
    ap.add_argument("--timeout", type=float, default=10.0, help="client gives up after this many seconds (counts as an error)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    queries = load_queries()
    rng = np.random.default_rng(0)
    limits = httpx.Limits(max_connections=2000, max_keepalive_connections=2000)
    async with httpx.AsyncClient(limits=limits) as client:
        h = (await client.get(args.url + "/health")).json()
        print(f"target {args.url}  products={h.get('products'):,}  mode={args.mode}  zipf={args.zipf}  {args.duration:.0f}s per rate\n")
        for q in queries[:3]:                                                  # warm up, uncounted
            await client.get(args.url + "/search", params={"q": q, "mode": args.mode}, timeout=60)
        print(f"{'rate/s':>7}{'sent':>6}{'ok':>6}{'err%':>7}{'thru/s':>8}{'p50 ms':>9}{'p95 ms':>9}{'p99 ms':>9}{'max ms':>9}")
        rows = []
        for rate in [float(x) for x in args.rates.split(",")]:
            r = await run_rate(client, args.url, queries, rate, args.duration, args.zipf, args.mode, args.timeout, rng)
            rows.append(r)
            def fmt(v):
                return f"{v:>9.0f}" if v is not None else f"{'-':>9}"

            print(f"{rate:>7.0f}{r['sent']:>6}{r['ok']:>6}{100 * r['errors'] / r['sent']:>6.1f}%{r['throughput']:>8.1f}"
                  f"{fmt(r['p50'])}{fmt(r['p95'])}{fmt(r['p99'])}{fmt(r['max'])}"
                  + (f"   errors: {r['error_codes']}" if r["error_codes"] else ""), flush=True)
            await asyncio.sleep(2)                                              # let the server drain between steps
    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
