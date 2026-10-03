"""
REST API for the product search engine -- built to survive heavy traffic.

    python product_search/api.py          # http://127.0.0.1:8000  (UI at /, interactive docs at /docs)

    GET  /search?q=remote+control+car&k=10&mode=hybrid_rerank&max_price=30&min_rating=4
    POST /search   {"q": "...", "k": 10, "mode": "...", "max_price": 30}
    GET  /health   GET /stats

What happens under load (the first version of this file just took a lock, and collapsed at ~3 req/s:
requests piled up, clients timed out, yet the server kept grinding through the abandoned ones):

  1. CACHE      identical recent searches are answered from memory (shopping traffic repeats heavily).
  2. ADMISSION  at most MAX_INFLIGHT requests are allowed inside at once; beyond that -> instant 503.
  3. SLOTS      only WORKERS searches run concurrently (the work is CPU-bound; more only adds contention).
  4. TIME BUDGET a request that waited longer than MAX_QUEUE_WAIT_MS for a slot is dropped with a 503
                instead of being processed after its client has already given up.
  5. DEGRADE    when the server is busy (>= DEGRADE_AT requests in flight) a hybrid_rerank search skips the
                slow reranker and answers with plain hybrid (flagged "degraded": true) -- worse ranking,
                several times faster, so the queue drains instead of growing.
All knobs are environment variables (see CONFIG below).
"""

import collections
import os
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))
from runtime import load_engine

CONFIG = {
    "DENSE": os.environ.get("SEARCH_DENSE", "hnsw" if (Path(__file__).parent / "index" / "hnsw.faiss").exists() else "flat"),
    "QUANTIZE_RERANKER": os.environ.get("SEARCH_QUANTIZE", "1") == "1",   # int8 reranker: ~1.6x faster, same nDCG (bench_stages.py)
    "RERANK_TOP": int(os.environ.get("SEARCH_RERANK_TOP", "25")),         # candidates the reranker re-reads (25: ~1/3 the time of 50)
    "WORKERS": int(os.environ.get("SEARCH_WORKERS", "3")),               # searches running at the same time
    "MAX_INFLIGHT": int(os.environ.get("SEARCH_MAX_INFLIGHT", "16")),    # admitted (running + waiting)
    "MAX_QUEUE_WAIT_MS": int(os.environ.get("SEARCH_MAX_QUEUE_WAIT_MS", "1500")),
    "DEGRADE_AT": int(os.environ.get("SEARCH_DEGRADE_AT", "6")),         # in-flight count at which reranking is skipped
    "CACHE_SIZE": int(os.environ.get("SEARCH_CACHE_SIZE", "20000")),
    "CACHE_TTL_S": int(os.environ.get("SEARCH_CACHE_TTL_S", "600")),
    "TORCH_THREADS": int(os.environ.get("SEARCH_TORCH_THREADS", "0")),   # 0 = leave PyTorch's default
}

state = {}
slots = threading.Semaphore(CONFIG["WORKERS"])
admission = threading.Lock()
stats = collections.Counter()
inflight = 0
recent_ms = collections.deque(maxlen=2000)             # server-side latency of recent successful searches
cache = collections.OrderedDict()                      # key -> (expires_at, response dict)
cache_lock = threading.Lock()
Mode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]


@asynccontextmanager
async def lifespan(app):
    if CONFIG["TORCH_THREADS"]:
        import torch
        torch.set_num_threads(CONFIG["TORCH_THREADS"])
    state["engine"] = load_engine(dense_kind=CONFIG["DENSE"], quantize_reranker=CONFIG["QUANTIZE_RERANKER"])
    import engine as engine_module
    engine_module.DEFAULTS["rerank_top"] = CONFIG["RERANK_TOP"]
    for m in ("hybrid_rerank", "hybrid"):          # pay one-off setup costs before the first real request
        state["engine"].search("warmup", k=3, mode=m)
    images = Path(__file__).parent / "data" / "images.parquet"       # optional: thumbnails for the web page
    if images.exists():
        import pandas as pd
        state["images"] = dict(pd.read_parquet(images).itertuples(index=False, name=None))
    print("Config:", CONFIG)
    yield


app = FastAPI(title="Toy Product Search", version="2.0", lifespan=lifespan)


class SearchRequest(BaseModel):
    q: str = Field(min_length=1, max_length=300)
    k: int = Field(10, ge=1, le=100)
    mode: Mode = "hybrid_rerank"
    min_price: float | None = Field(None, ge=0)
    max_price: float | None = Field(None, ge=0)
    min_rating: float | None = Field(None, ge=0, le=5)
    brand: str | None = Field(None, max_length=100)

    def cache_key(self):
        return (" ".join(self.q.lower().split()), self.k, self.mode, self.min_price, self.max_price,
                self.min_rating, (self.brand or "").lower())


def overloaded(reason, retry_after=1):
    return JSONResponse({"detail": reason}, status_code=503, headers={"Retry-After": str(retry_after)})


def cache_get(key):
    with cache_lock:
        hit = cache.get(key)
        if hit and hit[0] > time.monotonic():
            cache.move_to_end(key)
            return hit[1]
        if hit:
            del cache[key]
    return None


def cache_put(key, value):
    with cache_lock:
        cache[key] = (time.monotonic() + CONFIG["CACHE_TTL_S"], value)
        cache.move_to_end(key)
        while len(cache) > CONFIG["CACHE_SIZE"]:
            cache.popitem(last=False)


def run_search(req: SearchRequest):
    global inflight
    t_arrive = time.monotonic()
    if req.min_price is not None and req.max_price is not None and req.min_price > req.max_price:
        raise HTTPException(422, "min_price is greater than max_price")
    stats["requests"] += 1

    key = req.cache_key()
    hit = cache_get(key)
    if hit is not None:
        stats["cache_hits"] += 1
        return {**hit, "cached": True, "timings_ms": {"total": round((time.monotonic() - t_arrive) * 1000, 2)}}

    with admission:                                          # 2. admission control
        if inflight >= CONFIG["MAX_INFLIGHT"]:
            stats["shed_full"] += 1
            return overloaded("server busy: too many requests in flight")
        inflight += 1
        depth = inflight
    try:
        budget = CONFIG["MAX_QUEUE_WAIT_MS"] / 1000
        if not slots.acquire(timeout=budget):                # 3+4. wait for a run slot, but not forever
            stats["shed_timeout"] += 1
            return overloaded("server busy: waited too long for a free slot")
        try:
            waited_ms = (time.monotonic() - t_arrive) * 1000
            mode = req.mode
            degraded = False
            if mode == "hybrid_rerank" and depth >= CONFIG["DEGRADE_AT"]:    # 5. busy: skip the slow stage
                mode, degraded = "hybrid", True
                stats["degraded"] += 1
            resp = state["engine"].search(req.q, k=req.k, mode=mode, min_price=req.min_price, max_price=req.max_price,
                                          min_rating=req.min_rating, brand=req.brand)
        finally:
            slots.release()
    finally:
        with admission:
            inflight -= 1

    images = state.get("images", {})
    out = {
        "query": req.q, "mode": req.mode, "served_mode": mode, "degraded": degraded, "cached": False,
        "count": len(resp.results), "queue_wait_ms": round(waited_ms, 1), "timings_ms": resp.timings_ms,
        "results": [{**r.__dict__, "image": images.get(r.pid)} for r in resp.results],
    }
    recent_ms.append(resp.timings_ms.get("total", 0.0) + waited_ms)
    stats["ok"] += 1
    if not degraded:                                         # never cache a downgraded answer under the full-quality key
        cache_put(key, out)
    return out


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui():
    return (Path(__file__).parent / "ui.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok", "products": len(state["engine"].catalog)}


@app.get("/stats")
def get_stats():
    lat = np.array(recent_ms) if recent_ms else np.array([0.0])
    return {
        "config": CONFIG, "inflight": inflight, "cache_entries": len(cache), **stats,
        "cache_hit_rate": round(stats["cache_hits"] / max(stats["requests"], 1), 3),
        "recent_p50_ms": round(float(np.percentile(lat, 50)), 1), "recent_p95_ms": round(float(np.percentile(lat, 95)), 1),
    }


@app.get("/search")
def search_get(q: str = Query(..., min_length=1, max_length=300), k: int = Query(10, ge=1, le=100), mode: Mode = "hybrid_rerank",
               min_price: float | None = Query(None, ge=0), max_price: float | None = Query(None, ge=0),
               min_rating: float | None = Query(None, ge=0, le=5), brand: str | None = Query(None, max_length=100)):
    if q.strip() == "":
        raise HTTPException(422, "q must not be blank")
    return run_search(SearchRequest(q=q, k=k, mode=mode, min_price=min_price, max_price=max_price,
                                    min_rating=min_rating, brand=brand))


@app.post("/search")
def search_post(req: SearchRequest):
    if req.q.strip() == "":
        raise HTTPException(422, "q must not be blank")
    return run_search(req)


if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("SEARCH_HOST", "127.0.0.1"), port=int(os.environ.get("SEARCH_PORT", "8000")))
