"""
REST API for the product search engine.

    python product_search/api.py          # serves http://127.0.0.1:8000  (interactive docs at /docs)

    GET  /health
    GET  /search?q=remote+control+car&k=10&mode=hybrid_rerank&max_price=30&min_rating=4
    POST /search   {"q": "...", "k": 10, "mode": "...", "max_price": 30}

The models are not safe to call from several threads at once, and each search already uses
all CPU cores, so requests take turns (a lock). Throughput is therefore ~1 / latency per process;
to scale out, run more processes behind a load balancer.
"""

import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent))
from runtime import load_engine

state = {}
lock = threading.Lock()
Mode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]


@asynccontextmanager
async def lifespan(app):
    state["engine"] = load_engine()
    images = Path(__file__).parent / "data" / "images.parquet"       # optional: thumbnails for the web page
    if images.exists():
        import pandas as pd
        state["images"] = dict(pd.read_parquet(images).itertuples(index=False, name=None))
    state["engine"].search("warmup", k=3)          # pay one-off setup costs before the first real request
    yield


app = FastAPI(title="Toy Product Search", version="1.0", lifespan=lifespan)


class SearchRequest(BaseModel):
    q: str = Field(min_length=1, max_length=300)
    k: int = Field(10, ge=1, le=100)
    mode: Mode = "hybrid_rerank"
    min_price: Optional[float] = Field(None, ge=0)
    max_price: Optional[float] = Field(None, ge=0)
    min_rating: Optional[float] = Field(None, ge=0, le=5)
    brand: Optional[str] = Field(None, max_length=100)


def run_search(req: SearchRequest):
    if req.q.strip() == "":
        raise HTTPException(422, "q must not be blank")
    if req.min_price is not None and req.max_price is not None and req.min_price > req.max_price:
        raise HTTPException(422, "min_price is greater than max_price")
    with lock:
        resp = state["engine"].search(req.q, k=req.k, mode=req.mode, min_price=req.min_price,
                                      max_price=req.max_price, min_rating=req.min_rating, brand=req.brand)
    return {
        "query": req.q, "mode": req.mode, "count": len(resp.results), "timings_ms": resp.timings_ms,
        "results": [{**r.__dict__, "image": state.get("images", {}).get(r.pid)} for r in resp.results],
    }


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def ui():
    return (Path(__file__).parent / "ui.html").read_text(encoding="utf-8")


@app.get("/health")
def health():
    return {"status": "ok", "products": len(state["engine"].catalog)}


@app.get("/search")
def search_get(q: str = Query(..., min_length=1, max_length=300), k: int = Query(10, ge=1, le=100), mode: Mode = "hybrid_rerank",
               min_price: Optional[float] = Query(None, ge=0), max_price: Optional[float] = Query(None, ge=0),
               min_rating: Optional[float] = Query(None, ge=0, le=5), brand: Optional[str] = Query(None, max_length=100)):
    return run_search(SearchRequest(q=q, k=k, mode=mode, min_price=min_price, max_price=max_price,
                                    min_rating=min_rating, brand=brand))


@app.post("/search")
def search_post(req: SearchRequest):
    return run_search(req)


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
