# Design notes: product search

Decisions, the evidence behind them, and what went wrong along the way. Numbers come from the files in
[`product_search/results/`](../product_search/results) and can be reproduced with the scripts in
[`product_search/`](../product_search).

## 1. Why three retrieval stages instead of one

| Stage | Strength | Blind spot |
|---|---|---|
| Dense vectors (bge-small) | meaning: "toy for a 3 year old who loves trucks" | exact tokens: model numbers, rare names |
| BM25 keyword search | exact terms: `lego 75192` | synonyms and paraphrase |
| Cross-encoder reranker | reads query and product together, orders the best few | too slow to run on everything |

The averages tell a more modest story than the examples. Over 497 held-out queries, plain hybrid fusion did **not**
beat dense search (0.164 vs 0.166 nDCG@10); the reranker added roughly +0.01, and only the full tuned pipeline
was reliably better than dense alone (+0.016, 95% CI +0.006 to +0.025). I kept the three-stage design because the
failure modes are complementary and the cost is acceptable, but I report it as a small, real gain, not a leap.

## 2. Measuring honestly

* **Ground truth:** Amazon's ESCI shopping-query set (human Exact / Substitute / Complement / Irrelevant labels),
  restricted to US queries where at least half the labelled products are toys in my catalog. 2,272 queries:
  1,775 for tuning, 497 held out and never used for tuning.
* **Confidence intervals:** bootstrapped over queries, and paired differences between methods, so a gap is only
  called real when its interval excludes zero. Most gaps here are not.
* **Known bias:** ESCI only labels products Amazon's own search showed. A relevant toy that was never shown counts
  as irrelevant, and the catalog is full of near-duplicate listings, so absolute scores are low for every method.
  I trust the comparison between methods more than the absolute numbers.

## 3. Tuning without fooling myself

Fusion weights, the RRF constant, rerank depth and a score-blend weight were tuned on the **training** queries only
(`tune.py`). The grid's best cell used a rerank depth of 100; I shipped depth 50 (and later 25) because the quality
difference was within noise and the cost was not. The tuned settings improved the average slightly but moved a known
example (`lego star wars 75192`) from rank 1-2 down to rank 5-6; a follow-up breakdown on the 60 test queries that
contain numbers found no significant difference between tuned and untuned. That trade-off is documented, not hidden.

## 4. Making it fast: each change measured for speed *and* quality

| Change | Speed | Quality cost |
|---|---|---|
| Exact → HNSW dense search | 135 ms → 4 ms | finds 95% of the exact top-10; dense nDCG 0.1656 → 0.1606 |
| fp32 → int8 reranker | 386 ms → 229 ms (top 50) | none measurable (0.1824 → 0.1823) |
| Rerank top 50 → top 25 | 229 ms → 133 ms | 0.1823 → 0.1792 |

The first int8 attempt crashed: with sentence-transformers 6 the cross-encoder is a chain of modules, and replacing
the wrong attribute broke it. It is now quantized in place, after checking in isolation that scores still
correlate at 0.9996 with fp32.

## 5. The load test that broke the first version

The first API wrapped everything in one lock. At 3 requests/s, **70% of requests timed out; at 5/s, all of them**.
The cause was not slowness alone: the server kept working through requests whose clients had already given up, so
the backlog fed on itself. The fixes target exactly that:

* a **queue-time budget**: a request that waited too long is refused with an instant 503 instead of being served late;
* **bounded in-flight requests** and a **concurrency limit** matched to the CPU-bound work;
* **graceful degradation**: under load, skip the reranker (answers are flagged `degraded: true`);
* a **query cache**, because shopping traffic repeats heavily.

The load tester is **open-loop** (Poisson arrivals, latency measured from the scheduled send time), because a
closed-loop tester slows itself down when the server struggles and hides the overload.

Result on one laptop: 0% errors up to 12 req/s, then clean rejection instead of collapse. Honest limits: full-quality
(reranked) capacity is only about 5-8 req/s; above ~16 req/s most answers skip the reranker; the 71% cache hit rate in
the repeat-heavy test used a small, pre-warmed query pool and is a best case.

## 6. A bug only the container found

Running the Docker image against the real index, some searches returned HTTP 500 with
`Out of range float values are not JSON compliant: nan`. The image had installed pandas 3, which stores a missing
text field (a product with no brand) as NaN; the laptop's pandas 2 stored `None`. I had converted missing prices and
ratings but not missing brands or categories. The fix handles both pandas versions and has a regression test, and CI
now runs the tests under the newest pandas. Lesson: "works on my machine" is not evidence for a deployment.

## 7. What I would do next

* Fine-tune the embedding model on ESCI's training pairs (the largest untried quality lever).
* Per-client rate limiting and authentication before any public deployment (overload protection is global today).
* Measure on a dedicated Linux server: inside Docker on WSL2 the same code sustained about a third of bare-metal throughput.
* Judge relevance beyond ESCI's labelled set (e.g. sample unlabelled results and label them) to reduce the unjudged-item bias.
