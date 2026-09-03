"""
STEP 14: Now that training is fast (step 13), redo step 12's raw-vs-clean
comparison on the FULL real corpus (not a 500K-char sample) with a much
bigger merge count. This is the first time in this whole exercise we're
training at a scale where the resulting vocab can plausibly contain whole
common words, not just letter pairs.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import time
import importlib.util


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_here = os.path.dirname(os.path.abspath(__file__))
fast = load("fast_bpe", os.path.join(_here, "13_fast_bpe_train.py"))
codec = load("bpe_codec", os.path.join(_here, "04_encode_decode.py"))

DATA_DIR = os.path.join(_here, "data")
NUM_MERGES = 5000

with open(os.path.join(DATA_DIR, "corpus_raw.txt"), encoding="utf-8") as f:
    raw_text = f.read()
with open(os.path.join(DATA_DIR, "corpus_clean.txt"), encoding="utf-8") as f:
    clean_text = f.read()

print(f"RAW  : {len(raw_text):,} chars")
print(f"CLEAN: {len(clean_text):,} chars")
print(f"Training {NUM_MERGES} merges on each (full corpus, no truncation)...")
print()

t0 = time.time()
raw_merges, raw_vocab = fast.train_fast(raw_text, num_merges=NUM_MERGES, verbose=False)
print(f"RAW   trained in {time.time()-t0:.1f}s, vocab size {len(raw_vocab)}")

t0 = time.time()
clean_merges, clean_vocab = fast.train_fast(clean_text, num_merges=NUM_MERGES, verbose=False)
print(f"CLEAN trained in {time.time()-t0:.1f}s, vocab size {len(clean_vocab)}")
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

# how many merges are now FULL WHOLE WORDS (start with a space, rest all letters)?
def is_whole_word(b):
    t = b.decode("utf-8", errors="ignore")
    return t.startswith(" ") and t[1:].isalpha() and len(t) > 3

raw_words = sum(1 for nid in raw_merges.values() if is_whole_word(raw_vocab[nid]))
clean_words = sum(1 for nid in clean_merges.values() if is_whole_word(clean_vocab[nid]))
print(f"RAW   vocab: {raw_words}/{NUM_MERGES} merges are whole words (4+ letters, leading space)")
print(f"CLEAN vocab: {clean_words}/{NUM_MERGES} merges are whole words (4+ letters, leading space)")
print(f"Sample whole words learned (clean): "
      f"{[clean_vocab[nid].decode() for nid in list(clean_merges.values())[-30:] if is_whole_word(clean_vocab[nid])][:15]}")
print()

# held-out test sentences, spanning near-domain and off-domain text
test_sentences = [
    "The old house stood silent at the end of the lane, its windows dark "
    "against the evening sky, and no one dared to knock upon its door.",
    "I need to deploy the Kubernetes cluster before the sprint review.",
]

for text in test_sentences:
    raw_ids = codec.encode(text, raw_merges)
    clean_ids = codec.encode(text, clean_merges)
    b = len(text.encode("utf-8"))
    print(f"text: {text!r}")
    print(f"  RAW-trained  : {len(raw_ids):>3} tokens ({b/len(raw_ids):.2f}x)")
    print(f"  CLEAN-trained: {len(clean_ids):>3} tokens ({b/len(clean_ids):.2f}x)")
    print()

print(f"At {NUM_MERGES} merges (vs step 12's 600), both vocabs are big enough to")
print("capture real whole words -- this is the scale where 'training data quality'")
print("stops being about a handful of junk tokens and starts being about HOW MANY")
print("of your limited merge slots go to genuinely useful, generalizable words")
print("versus low-value structural noise that only exists because of the source")
print("format (chapter headers, illustration captions), which is exactly what")
print("real pretraining pipelines are trying to maximize when they clean data at scale.")
