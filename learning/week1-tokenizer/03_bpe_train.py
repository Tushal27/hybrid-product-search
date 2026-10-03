"""
STEP 3: The actual Byte-Pair Encoding (BPE) training algorithm.

Idea: start with raw bytes (ids 0-255) as your vocab. Repeatedly find the
PAIR of adjacent token ids that occurs most often across the whole corpus,
and merge that pair into ONE new token id. Repeat N times. Each merge
grows the vocab by exactly 1 and shrinks the average sequence length.
That's it -- that's the entire algorithm. GPT-4's real vocab is just the
result of running this loop ~100,000 times over trillions of bytes.

Critical detail we already earned in step 2: merges must NEVER cross a
regex pre-tokenization chunk boundary. So we keep the corpus as a LIST of
chunks (each its own list of byte ids), and only ever merge pairs that are
adjacent *within* a chunk.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import regex
from collections import Counter

GPT4_SPLIT_PATTERN = (
    r"'(?i:[sdmt]|ll|ve|re)"
    r"|[^\r\n\p{L}\p{N}]?+\p{L}++"
    r"|\p{N}{1,3}+"
    r"| ?[^\s\p{L}\p{N}]++[\r\n]*+"
    r"|\s++$"
    r"|\s*[\r\n]"
    r"|\s+(?!\S)"
    r"|\s"
)
compiled_pattern = regex.compile(GPT4_SPLIT_PATTERN)

# A real (if small) corpus: openings of several public-domain novels.
# Enough natural English repetition ("the", " the", "and", "was", "of") to
# see BPE build sensible merges, without needing a network download.
CORPUS = """
It was the best of times, it was the worst of times, it was the age of
wisdom, it was the age of foolishness, it was the epoch of belief, it was
the epoch of incredulity, it was the season of Light, it was the season
of Darkness, it was the spring of hope, it was the winter of despair.

It is a truth universally acknowledged, that a single man in possession
of a good fortune, must be in want of a wife. However little known the
feelings or views of such a man may be on his first entering a
neighbourhood, this truth is so well fixed in the minds of the
surrounding families, that he is considered the rightful property of
some one or other of their daughters.

Call me Ishmael. Some years ago, never mind how long precisely, having
little or no money in my purse, and nothing particular to interest me on
shore, I thought I would sail about a little and see the watery part of
the world.

To Sherlock Holmes she is always the woman. I have seldom heard him
mention her under any other name. In his eyes she eclipses and
predominates the whole of her sex.
"""


def get_chunk_stats(chunks: list[list[int]]) -> Counter:
    """Count how often each adjacent pair (id_i, id_i+1) occurs,
    only counting pairs INSIDE a chunk, never across chunk boundaries."""
    counts = Counter()
    for ids in chunks:
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
    return counts


def merge_chunk(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Replace every occurrence of `pair` in `ids` with `new_id`."""
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


def train(text: str, num_merges: int, verbose: bool = True):
    # 1. pre-tokenize with the GPT-4 regex (never merge across these boundaries)
    text_chunks = compiled_pattern.findall(text)
    # 2. each chunk -> its raw utf-8 byte ids
    chunks = [list(ch.encode("utf-8")) for ch in text_chunks]

    merges: dict[tuple[int, int], int] = {}          # (id1, id2) -> new_id
    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}  # id -> bytes

    for i in range(num_merges):
        stats = get_chunk_stats(chunks)
        if not stats:
            break
        # tie-break explicitly on the pair itself (not dict iteration order) so
        # this is reproducible regardless of how `stats` was built -- matters
        # for step 13's incremental trainer to be able to match this exactly.
        top_pair = max(stats, key=lambda p: (stats[p], p))
        new_id = 256 + i
        chunks = [merge_chunk(c, top_pair, new_id) for c in chunks]
        merges[top_pair] = new_id
        vocab[new_id] = vocab[top_pair[0]] + vocab[top_pair[1]]
        if verbose:
            print(
                f"merge {i+1:>3}: {top_pair} -> {new_id}   "
                f"({vocab[top_pair[0]]!r} + {vocab[top_pair[1]]!r} = {vocab[new_id]!r})"
                f"   [{stats[top_pair]} occurrences]"
            )

    return merges, vocab


if __name__ == "__main__":
    merges, vocab = train(CORPUS, num_merges=20, verbose=True)

    print()
    print(f"Started with 256 byte tokens, learned {len(merges)} merges.")
    print(f"Final vocab size: {len(vocab)}")
    print()
    print("Look at the merges above: early merges are tiny (letter pairs like")
    print("'t'+'h' -> 'th'), later merges combine those into whole words like")
    print("' the', ' of', ' was'. This bottom-up growth from bytes -> subwords")
    print("-> whole words is the entire trick behind BPE. GPT-4 just does this")
    print("~100,000 times on trillions of bytes instead of 20 times on one page.")
