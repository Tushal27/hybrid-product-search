"""
STEP 4: Use the merges we learned to actually tokenize NEW text (encode),
and turn token ids back into text (decode).

ENCODE:
  1. Split new text into chunks with the same regex (never merge across
     chunk boundaries -- has to match training exactly).
  2. Convert each chunk to raw byte ids.
  3. Repeatedly find the pair in this chunk that has the EARLIEST-LEARNED
     merge (lowest new_id) and apply it, until no known merge pair remains.
     (Must apply merges in the order they were learned -- e.g. you can't
     merge ' the' before 'th' exists, so lowest-id-first replays training
     order correctly.)

DECODE:
  Just look up each id's bytes in vocab, concatenate, decode as utf-8.
  Since every token is ultimately just a concatenation of raw bytes, this
  round-trips PERFECTLY for any valid text -- no information is ever lost,
  unlike character-vocab tokenizers that hit <unk> on unseen characters.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import importlib.util

# 03_bpe_train.py starts with a digit, so it can't be imported with a plain
# `import` statement -- load it directly from its file path instead.
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("bpe_train", os.path.join(_here, "03_bpe_train.py"))
bpe_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bpe_train)

compiled_pattern = bpe_train.compiled_pattern
get_chunk_stats = bpe_train.get_chunk_stats
merge_chunk = bpe_train.merge_chunk
CORPUS = bpe_train.CORPUS


def encode_chunk(ids: list[int], merges: dict[tuple[int, int], int]) -> list[int]:
    while len(ids) >= 2:
        stats = get_chunk_stats([ids])
        # among pairs present in this chunk, pick the one learned EARLIEST
        # during training (lowest merge id) -- that replays training order.
        pair = min(stats, key=lambda p: merges.get(p, float("inf")))
        if pair not in merges:
            break
        ids = merge_chunk(ids, pair, merges[pair])
    return ids


def encode(text: str, merges: dict[tuple[int, int], int]) -> list[int]:
    text_chunks = compiled_pattern.findall(text)
    ids = []
    for chunk in text_chunks:
        ids.extend(encode_chunk(list(chunk.encode("utf-8")), merges))
    return ids


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    raw_bytes = b"".join(vocab[i] for i in ids)
    return raw_bytes.decode("utf-8", errors="replace")


if __name__ == "__main__":
    # train a slightly bigger vocab than step 3 so we get whole common words
    merges, vocab = bpe_train.train(CORPUS, num_merges=80, verbose=False)
    print(f"Trained {len(merges)} merges. Vocab size: {len(vocab)}")
    print()

    test_sentences = [
        "It was the age of wisdom.",           # close to training data
        "The wife has a fortune.",              # new sentence, familiar words
        "I love debugging tokenizers 🚀",        # unseen words + emoji
        "日本語のテスト",                          # completely different script
    ]

    for text in test_sentences:
        ids = encode(text, merges)
        decoded = decode(ids, vocab)
        raw_byte_len = len(text.encode("utf-8"))
        print(f"text        : {text!r}")
        print(f"token ids   : {ids}")
        print(f"num tokens  : {len(ids)}   (vs {raw_byte_len} raw bytes -> "
              f"{raw_byte_len/len(ids):.2f}x compression)")
        print(f"decoded     : {decoded!r}")
        print(f"round-trip OK: {decoded == text}")
        assert decoded == text, "LOSSY ROUND TRIP -- bug!"
        print()

    print("Every single one round-trips perfectly, including the emoji and")
    print("Japanese text that never appeared ANYWHERE in training. That's the")
    print("byte-level guarantee from step 1 paying off: unseen chunks just")
    print("fall back to raw byte tokens (ids 0-255), never an <unk>.")
