"""
STEP 13: Fix the performance problem from step 12.

The naive trainer (step 3) does, every single merge:
  1. rescan EVERY token in the corpus to count all adjacent pairs -- O(N)
  2. rescan EVERY token again to apply the merge -- O(N)
That's O(N * num_merges) total. At 500K chars x 600 merges, ~1 minute.
At GPT-4 scale (trillions of tokens x 100k merges) this is simply impossible.

Two real optimizations (this is genuinely how HuggingFace's `tokenizers`
and Google's `sentencepiece` do it, not a toy simplification):

  (A) COLLAPSE DUPLICATE CHUNKS. After the regex pre-split, the word " the"
      might appear 30,000 times in a corpus, but it's the SAME bytes every
      time. Instead of storing 30,000 copies, store {" the": 30000} and do
      all counting/merging ONCE per unique chunk, weighted by its count.
      For natural-language text this alone can shrink the working set by
      1-2 orders of magnitude (a few thousand unique words vs millions of
      word occurrences).

  (B) INCREMENTAL PAIR COUNTS. Maintain a running pair -> count dict and a
      pair -> {chunk ids containing it} index. When you merge a pair, ONLY
      the chunks that actually contain that pair need their counts updated
      -- everything else in the corpus is provably unaffected and can be
      skipped entirely. Finding the next merge is then a max() over the
      (much smaller) set of *distinct* pairs, not a rescan of all tokens.

Together: total work becomes roughly proportional to how often each
merged pair actually occurs, summed over all merges -- not
corpus_size * num_merges.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import time
import importlib.util
from collections import Counter, defaultdict

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("bpe_train", os.path.join(_here, "03_bpe_train.py"))
bpe_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bpe_train)

compiled_pattern = bpe_train.compiled_pattern
merge_chunk = bpe_train.merge_chunk  # unchanged, reused as-is


class _Chunk:
    __slots__ = ("ids", "count")
    def __init__(self, ids, count):
        self.ids = ids
        self.count = count


def train_fast(text: str, num_merges: int, verbose: bool = False):
    # (A) collapse to unique chunks with frequency counts
    text_chunks = compiled_pattern.findall(text)
    chunk_freq = Counter(text_chunks)
    chunks: list[_Chunk] = [
        _Chunk(list(s.encode("utf-8")), freq) for s, freq in chunk_freq.items()
    ]

    merges: dict[tuple[int, int], int] = {}
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}

    # (B) build initial pair counts + pair -> chunk-index reverse index ONCE
    pair_counts: Counter = Counter()
    pair_to_chunks: dict[tuple[int, int], set] = defaultdict(set)
    for ci, chunk in enumerate(chunks):
        for pair in zip(chunk.ids, chunk.ids[1:]):
            pair_counts[pair] += chunk.count
            pair_to_chunks[pair].add(ci)

    for i in range(num_merges):
        if not pair_counts:
            break
        top_pair = max(pair_counts, key=lambda p: (pair_counts[p], p))
        new_id = 256 + i
        count_before = pair_counts[top_pair]

        # only touch chunks that actually contain top_pair -- everything
        # else in the corpus is untouched by this merge, full stop.
        affected = list(pair_to_chunks.get(top_pair, ()))
        for ci in affected:
            chunk = chunks[ci]
            old_ids = chunk.ids
            # remove this chunk's old pair contributions
            for p in zip(old_ids, old_ids[1:]):
                pair_counts[p] -= chunk.count
                if pair_counts[p] <= 0:
                    del pair_counts[p]
                pair_to_chunks[p].discard(ci)

            new_ids = merge_chunk(old_ids, top_pair, new_id)
            chunk.ids = new_ids

            # add this chunk's new pair contributions
            for p in zip(new_ids, new_ids[1:]):
                pair_counts[p] += chunk.count
                pair_to_chunks[p].add(ci)

        merges[top_pair] = new_id
        vocab[new_id] = vocab[top_pair[0]] + vocab[top_pair[1]]
        if verbose:
            print(f"merge {i+1:>4}: {top_pair} -> {new_id}  "
                  f"({vocab[top_pair[0]]!r}+{vocab[top_pair[1]]!r}={vocab[new_id]!r})  "
                  f"[{count_before} occurrences, touched {len(affected)} unique chunks]")

    return merges, vocab


if __name__ == "__main__":
    # ---------- 1. CORRECTNESS: must match the naive trainer exactly ----------
    print("Verifying train_fast() produces IDENTICAL output to naive train()...")
    naive_merges, naive_vocab = bpe_train.train(bpe_train.CORPUS, num_merges=80, verbose=False)
    fast_merges, fast_vocab = train_fast(bpe_train.CORPUS, num_merges=80, verbose=False)
    assert naive_merges == fast_merges, "MISMATCH in merges -- fast version has a bug!"
    assert naive_vocab == fast_vocab, "MISMATCH in vocab -- fast version has a bug!"
    print("MATCH: identical merges and vocab, 80/80.")
    print()

    # ---------- 2. SPEED on the real corpus from step 10-11 ----------
    data_dir = os.path.join(_here, "data")
    with open(os.path.join(data_dir, "corpus_clean.txt"), encoding="utf-8") as f:
        real_text = f.read()

    for sample_chars, n_merges in [(500_000, 600), (len(real_text), 2000)]:
        sample = real_text[:sample_chars]
        print(f"--- {sample_chars:,} chars, {n_merges} merges ---")

        t0 = time.time()
        _, _ = train_fast(sample, num_merges=n_merges, verbose=False)
        fast_time = time.time() - t0
        print(f"  train_fast : {fast_time:.2f}s")

        if sample_chars <= 500_000:
            t0 = time.time()
            _, _ = bpe_train.train(sample, num_merges=n_merges, verbose=False)
            naive_time = time.time() - t0
            print(f"  naive train: {naive_time:.2f}s")
            print(f"  speedup    : {naive_time/fast_time:.1f}x")
        else:
            print(f"  (skipping naive train here -- at this size it would take way too long)")
        print()
