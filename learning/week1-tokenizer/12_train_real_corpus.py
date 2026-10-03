"""
STEP 12: Train BPE (unchanged from step 3) on real raw vs real cleaned data,
at a scale where the training time is actually noticeable (unlike our
sub-second toy corpus). This is the same exercise as step 9, just on real
downloaded books instead of hand-written synthetic junk.

Our trainer recomputes pair counts over the WHOLE corpus every merge (see
the benchmark comment below) -- that's O(corpus_size * num_merges), which
is why we sample instead of using the full ~3.8M-character corpus. Real
BPE trainers (sentencepiece, tiktoken's own trainer) use incremental
count updates instead of full rescans, which is a real engineering
optimization we're intentionally not building here -- this script would
take way too long on the full corpus with our simple version, which is
itself an honest lesson: the naive algorithm you learn conceptually is
rarely the one that ships to production unmodified.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import importlib.util

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("bpe_train", os.path.join(_here, "03_bpe_train.py"))
bpe_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bpe_train)

DATA_DIR = os.path.join(_here, "data")
SAMPLE_CHARS = 500_000
NUM_MERGES = 600

with open(os.path.join(DATA_DIR, "corpus_raw.txt"), encoding="utf-8") as f:
    raw_text = f.read()[:SAMPLE_CHARS]
with open(os.path.join(DATA_DIR, "corpus_clean.txt"), encoding="utf-8") as f:
    clean_text = f.read()[:SAMPLE_CHARS]

print(f"Training on {SAMPLE_CHARS:,}-char samples, {NUM_MERGES} merges each "
      f"(this takes roughly a minute per run)...")
print()

import time
t0 = time.time()
raw_merges, raw_vocab = bpe_train.train(raw_text, num_merges=NUM_MERGES, verbose=False)
print(f"RAW   corpus trained in {time.time()-t0:.1f}s, vocab size {len(raw_vocab)}")

t0 = time.time()
clean_merges, clean_vocab = bpe_train.train(clean_text, num_merges=NUM_MERGES, verbose=False)
print(f"CLEAN corpus trained in {time.time()-t0:.1f}s, vocab size {len(clean_vocab)}")
print()

junk_markers = ["chapter", "illustration", "gutenberg", "footnote"]
def looks_like_junk(b):
    t = b.decode("utf-8", errors="ignore").lower()
    return any(m in t for m in junk_markers)

raw_junk = [raw_vocab[nid] for nid in raw_merges.values() if looks_like_junk(raw_vocab[nid])]
clean_junk = [clean_vocab[nid] for nid in clean_merges.values() if looks_like_junk(clean_vocab[nid])]
print(f"RAW   vocab: {len(raw_junk)}/{NUM_MERGES} merges look like structural junk: {raw_junk}")
print(f"CLEAN vocab: {len(clean_junk)}/{NUM_MERGES} merges look like structural junk: {clean_junk}")
print()

# held-out test: a sentence NOT drawn from any of the 6 books, to compare
# real compression on genuinely unseen (but similar-genre) prose
test_sentence = (
    "The old house stood silent at the end of the lane, its windows dark "
    "against the evening sky, and no one dared to knock upon its door."
)


def encode_with(text, merges):
    import os as _os
    _spec2 = importlib.util.spec_from_file_location("bpe_codec", _os.path.join(_here, "04_encode_decode.py"))
    codec = importlib.util.module_from_spec(_spec2)
    _spec2.loader.exec_module(codec)
    return codec.encode(text, merges), codec


ids_raw, codec = encode_with(test_sentence, raw_merges)
ids_clean, _ = encode_with(test_sentence, clean_merges)
raw_bytes = len(test_sentence.encode("utf-8"))

print(f"Held-out sentence: {test_sentence!r}")
print(f"  raw bytes         : {raw_bytes}")
print(f"  RAW-trained vocab : {len(ids_raw)} tokens  ({raw_bytes/len(ids_raw):.2f}x compression)")
print(f"  CLEAN-trained vocab: {len(ids_clean)} tokens  ({raw_bytes/len(ids_clean):.2f}x compression)")
