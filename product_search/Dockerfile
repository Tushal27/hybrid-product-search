# Toy product search API.   Build:  docker build -t toy-search product_search
#                           Run:    see product_search/DEPLOY.md
# Works on x86-64 and ARM64 servers (python:3.12-slim is multi-arch; every dependency ships wheels for both).
FROM python:3.12-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/hf_cache

COPY requirements-serve.txt .
# CPU-only PyTorch (the default PyPI build for Linux drags in ~2.5 GB of CUDA libraries we never use)
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r requirements-serve.txt

# Bake the two small models (~220 MB) into the image, so the server never needs internet at run time
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"
ENV HF_HUB_OFFLINE=1

COPY api.py engine.py runtime.py catalog.py embedder.py ui.html tuned_settings.json ./

# Mounted from the host:  /app/index (hnsw.faiss, bm25/)   /app/data (catalog.parquet, images.parquet)
ENV SEARCH_HOST=0.0.0.0 SEARCH_PORT=8000
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).status == 200 else 1)"
# ONE process on purpose: the index (~4 GB RAM) and cache live in memory; more processes would multiply the RAM.
CMD ["python", "api.py"]
