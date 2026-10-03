"""
STEP 5: The real test. Load GPT-4's ACTUAL trained merges (cl100k_base,
same tokenizer used by GPT-3.5-turbo and GPT-4) into OUR OWN encode/decode
engine from steps 3-4, and check we produce byte-for-byte identical token
ids to the real `tiktoken` library on a range of tricky inputs.

If this matches, it proves our understanding of the ALGORITHM is complete
and correct -- the only thing we didn't do ourselves is the multi-trillion-
token training run that produced the merge table.

One gotcha tiktoken does that we haven't seen yet: cl100k_base's 256
single-byte tokens are NOT id-equal to their byte value (rank of byte 0x00
is not 0). This is a historical quirk of how OpenAI built the vocab. So we
need a "byte shuffle" step: raw byte value -> shuffled id, before doing any
merging, and the inverse when decoding.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import importlib.util
import tiktoken

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("bpe_train", os.path.join(_here, "03_bpe_train.py"))
bpe_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bpe_train)
_spec2 = importlib.util.spec_from_file_location("bpe_codec", os.path.join(_here, "04_encode_decode.py"))
bpe_codec = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(bpe_codec)

compiled_pattern = bpe_train.compiled_pattern
get_chunk_stats = bpe_train.get_chunk_stats
merge_chunk = bpe_train.merge_chunk


def bpe_apply(mergeable_ranks: dict[bytes, int], token: bytes, max_rank: int | None) -> list[bytes]:
    """Given raw bytes of ONE token, replay merges (in rank order) up to
    (but not including) max_rank. This is how we reverse-engineer the pair
    that produced each multi-byte token, since tiktoken only stores the
    final byte-string -> rank table, not the merge sequence itself."""
    parts = [bytes([b]) for b in token]
    while True:
        min_idx, min_rank = None, None
        for i, (a, b) in enumerate(zip(parts[:-1], parts[1:])):
            rank = mergeable_ranks.get(a + b)
            if rank is not None and (min_rank is None or rank < min_rank):
                min_idx, min_rank = i, rank
        if min_rank is None or (max_rank is not None and min_rank >= max_rank):
            break
        parts = parts[:min_idx] + [parts[min_idx] + parts[min_idx + 1]] + parts[min_idx + 2:]
    return parts


def recover_merges(mergeable_ranks: dict[bytes, int]) -> dict[tuple[int, int], int]:
    """Reconstruct the (id1, id2) -> new_id merge table from tiktoken's
    public bytes->rank table alone."""
    merges = {}
    for token, rank in mergeable_ranks.items():
        if len(token) == 1:
            continue
        pair = bpe_apply(mergeable_ranks, token, max_rank=rank)
        assert len(pair) == 2, f"expected a pair, got {pair} for {token!r}"
        id0, id1 = mergeable_ranks[pair[0]], mergeable_ranks[pair[1]]
        merges[(id0, id1)] = rank
    return merges


class OurGPT4Tokenizer:
    """Our own encode/decode engine (from steps 3-4), loaded with GPT-4's
    real trained merges instead of our toy corpus's merges."""

    def __init__(self):
        enc = tiktoken.get_encoding("cl100k_base")
        mergeable_ranks = enc._mergeable_ranks
        self.merges = recover_merges(mergeable_ranks)
        self.vocab = {rank: tok for tok, rank in mergeable_ranks.items()}
        # byte shuffle: raw byte value -> its (non-identity!) rank
        self.byte_shuffle = {i: mergeable_ranks[bytes([i])] for i in range(256)}
        self.inverse_byte_shuffle = {v: k for k, v in self.byte_shuffle.items()}

    def encode(self, text: str) -> list[int]:
        ids = []
        for chunk in compiled_pattern.findall(text):
            chunk_bytes = chunk.encode("utf-8")
            shuffled = [self.byte_shuffle[b] for b in chunk_bytes]
            ids.extend(bpe_codec.encode_chunk(shuffled, self.merges))
        return ids

    def decode(self, ids: list[int]) -> str:
        raw = b"".join(self.vocab[i] for i in ids)
        return raw.decode("utf-8", errors="replace")


if __name__ == "__main__":
    ours = OurGPT4Tokenizer()
    ref = tiktoken.get_encoding("cl100k_base")

    test_cases = [
        "Hello, world!",
        "don't stop believin', we're almost there",
        "The 2024 model costs $1,234,567.89 -- wow!",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "日本語のテキストも大丈夫ですか?",
        "I love debugging tokenizers 🚀🔥 so much!!!",
        "supercalifragilisticexpialidocious",
        "   leading and trailing whitespace   ",
    ]

    all_match = True
    for text in test_cases:
        ours_ids = ours.encode(text)
        real_ids = ref.encode(text)
        match = ours_ids == real_ids
        all_match &= match
        status = "MATCH" if match else "MISMATCH"
        print(f"[{status}] {text!r}")
        print(f"  ours : {ours_ids}")
        print(f"  real : {real_ids}")
        # also check our decode reconstructs the original text
        assert ours.decode(ours_ids) == text
        print()

    print("=" * 60)
    if all_match:
        print("ALL TEST CASES MATCH tiktoken's real cl100k_base output.")
        print("Our from-scratch algorithm (regex split + byte shuffle +")
        print("lowest-rank-first BPE merging) is a byte-for-byte faithful")
        print("reimplementation of what powers GPT-3.5 / GPT-4 tokenization.")
    else:
        print("Some mismatches above -- worth digging into why.")
