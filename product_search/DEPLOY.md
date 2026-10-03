# Deploying the toy product search to a cloud VM

Everything below is manual on purpose: it creates accounts and opens a server to the internet, so you
do those steps yourself. Nothing here has been run on a real cloud VM yet -- only the local pieces were tested.

## What the server needs

| | |
|---|---|
| RAM | the server process uses ~4.2 GB; plan for **6-8 GB** (1 GB "free tier" micro-VMs will not work) |
| Disk | ~2.2 GB of index/data + ~3 GB for the Docker image |
| CPU | any modern CPU; more cores = more concurrent searches. x86-64 or ARM64 both work |
| Upload | **~2.2 GB** from your laptop (files below) |

Files to upload (everything else is rebuilt or baked into the image):

```
product_search/index/hnsw.faiss      1.6 GB   (vectors are inside this file; vectors.npy is NOT needed)
product_search/index/bm25/           0.25 GB
product_search/data/catalog.parquet  0.27 GB
product_search/data/images.parquet   0.02 GB   (optional: thumbnails in the web page)
```

## "Free" options -- read this before choosing

Free tiers change; verify current terms on the provider's site.

* **Oracle Cloud "Always Free" ARM VM** (Ampere A1) is the only free offer I know of with enough memory
  (its allowance has been up to 4 cores / 24 GB RAM). Catches: it needs a credit card for identity
  verification; capacity in popular regions is often "out of stock" (retry or pick another region);
  Oracle has reclaimed idle Always Free instances; the cores are slower than a desktop CPU, so expect
  **lower throughput than the laptop numbers** -- re-run the load test on it.
  The libraries used all publish ARM64 wheels, but the int8 reranker has not been tried on ARM here.
* **AWS / Google / Azure free tiers** give 1 GB RAM machines: too small. Their new-account *credits*
  (e.g. $300 for 90 days on Google Cloud) can pay for a 8 GB VM for a while, but that is a trial, not free forever.
* **Your own laptop/PC shared through a tunnel** (Cloudflare Tunnel, ngrok) is free and fine for a demo,
  but it is only up while your machine is on.

## Steps (generic Linux VM, Ubuntu 22.04/24.04)

1. **Create the VM** (8 GB+ RAM, Ubuntu). Note its public IP. In the provider's *network/security rules*
   allow inbound TCP **8000** (or 80/443 if you add a reverse proxy). On Oracle Ubuntu images the VM's own
   firewall also blocks ports: `sudo iptables -I INPUT -p tcp --dport 8000 -j ACCEPT` (and persist it).
2. **Install Docker** on the VM: `curl -fsSL https://get.docker.com | sudo sh`
3. **Copy the project code and data** from your laptop (PowerShell; replace USER and IP):
   ```
   scp -r product_search\*.py product_search\ui.html product_search\tuned_settings.json product_search\requirements-serve.txt product_search\Dockerfile product_search\.dockerignore USER@IP:~/search/
   scp product_search\index\hnsw.faiss USER@IP:~/search/index/
   scp -r product_search\index\bm25 USER@IP:~/search/index/
   scp product_search\data\catalog.parquet product_search\data\images.parquet USER@IP:~/search/data/
   ```
   (create the folders first: `ssh USER@IP "mkdir -p ~/search/index ~/search/data"`). The 1.6 GB file dominates;
   on a slow upload connection this is the long step. `rsync -P` can resume an interrupted upload.
4. **Build and run** on the VM:
   ```
   cd ~/search
   docker build -t toy-search .
   docker run -d --name search --restart unless-stopped -p 8000:8000 \
     -v ~/search/index:/app/index:ro -v ~/search/data:/app/data:ro \
     toy-search
   docker logs -f search        # wait for "Engine ready"; loading takes ~20-60 s
   ```
5. **Check it**: `curl http://IP:8000/health`, open `http://IP:8000/` in a browser, and look at `/stats`.
6. **Load test it from your laptop** (network latency is then included, which is realistic):
   ```
   python product_search/loadtest.py --url http://IP:8000 --rates 2,5,10,20 --duration 30 --zipf 1.0
   ```

## Tuning on the VM (environment variables for `docker run -e ...`)

| Variable | Default | When to change |
|---|---|---|
| `SEARCH_WORKERS` | 3 | searches running at once; set near the number of CPU cores / 2 |
| `SEARCH_TORCH_THREADS` | PyTorch default | set to the core count if CPU use looks low |
| `SEARCH_RERANK_TOP` | 25 | 50 = best quality, ~1.7x slower; 15 = faster, slightly worse |
| `SEARCH_DEGRADE_AT` | 6 | in-flight requests at which the reranker is skipped to protect latency |
| `SEARCH_MAX_QUEUE_WAIT_MS` | 1500 | how long a request may wait for a slot before an instant 503 |
| `SEARCH_MAX_INFLIGHT` | 16 | requests admitted at once; more get an instant 503 |
| `SEARCH_CACHE_SIZE` / `_TTL_S` | 20000 / 600 | query cache |

## Before exposing it to the public internet

This service has **no authentication and no per-visitor rate limit**. Its overload protection is global:
one abusive client can use up the whole capacity and everyone else gets 503s. At minimum put a reverse
proxy (Caddy or nginx) in front for HTTPS and per-IP rate limiting (`limit_req` in nginx), and keep the
port 8000 closed to the world once the proxy is in place. The page links out to amazon.com product pages
and loads thumbnails from Amazon's image servers; the data comes from public research datasets, so check
their terms before using it commercially.

## Capacity you can expect (measured on the dev laptop, one process, load generator on the same machine)

* full-quality hybrid + rerank: ~5-8 searches/s
* under heavier load the server skips the reranker (answers flagged `"degraded": true`): ~20-25 searches/s
* beyond that it answers `503` immediately instead of slowing down
* repeated queries are served from the cache in ~20 ms
Numbers on a cloud VM will differ -- measure with step 6.
