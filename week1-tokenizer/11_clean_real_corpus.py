"""
STEP 11: Run the real cleaning pipeline on the real 3.9MB / 12,865-paragraph
corpus from step 10.

New problem we didn't hit with our tiny synthetic examples: step 7's near-
dedup compared every document to every other kept document (O(n^2)). At
12,865 paragraphs that's ~165 MILLION pairwise comparisons -- genuinely too
slow. This is the exact wall real pipelines hit, and the reason they use
LSH (Locality-Sensitive Hashing) instead of brute force.

LSH idea: split each document's MinHash signature (64 numbers) into bands
of a few numbers each (say, 16 bands of 4 numbers). Two documents are only
COMPARED if they share an identical band somewhere -- i.e. we hash each
band to a bucket and only check documents that land in the same bucket.
Similar documents are likely to match in at least one band by chance;
dissimilar documents are very unlikely to match in ANY band. This turns
"compare everything to everything" into "compare only plausible matches,"
which is what makes near-dedup tractable at billions of documents.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import re
import glob
import hashlib
from collections import defaultdict

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

# ---------- load + split into paragraph-level documents ----------
documents = []  # list of (text, source_book)
for path in sorted(glob.glob(os.path.join(DATA_DIR, "*.txt"))):
    name = os.path.splitext(os.path.basename(path))[0]
    with open(path, encoding="utf-8") as f:
        text = f.read()
    for para in text.split("\n\n"):
        para = para.strip()
        if para:
            documents.append((para, name))

print(f"Loaded {len(documents)} paragraphs from {len(set(b for _, b in documents))} books.")
print()

# ---------- exact dedup ----------
def normalize(text): return " ".join(text.split()).lower()

def exact_dedup(docs):
    seen, kept = set(), []
    for text, src in docs:
        h = hashlib.sha256(normalize(text).encode()).hexdigest()
        if h not in seen:
            seen.add(h); kept.append((text, src))
    return kept

after_exact = exact_dedup(documents)
print(f"After exact dedup   : {len(after_exact)}  "
      f"(dropped {len(documents) - len(after_exact)} byte-identical repeats "
      f"-- e.g. bare 'CHAPTER I.' style headers repeating across books)")

# ---------- near dedup via LSH banding (scales to real sizes) ----------
NUM_HASHES = 64
NUM_BANDS = 16
ROWS_PER_BAND = NUM_HASHES // NUM_BANDS  # 4

def shingles(text, k=3):
    words = normalize(text).split()
    if len(words) < k:
        return {tuple(words)}
    return {tuple(words[i:i + k]) for i in range(len(words) - k + 1)}

def minhash_sig(shingle_set):
    if not shingle_set:
        return [0] * NUM_HASHES
    return [min(int(hashlib.md5(f"{seed}:{s}".encode()).hexdigest(), 16) for s in shingle_set)
            for seed in range(NUM_HASHES)]

def sig_similarity(a, b):
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)

def lsh_near_dedup(docs, threshold=0.5):
    kept = []
    kept_sigs = []
    # buckets[band_index][band_value] -> list of indices into `kept`
    buckets = [defaultdict(list) for _ in range(NUM_BANDS)]
    dropped_examples = []

    for text, src in docs:
        sig = minhash_sig(shingles(text))
        bands = [tuple(sig[b * ROWS_PER_BAND:(b + 1) * ROWS_PER_BAND]) for b in range(NUM_BANDS)]

        # gather candidates: any kept doc sharing ANY band bucket with this one
        candidate_idxs = set()
        for b, band_val in enumerate(bands):
            candidate_idxs.update(buckets[b].get(band_val, []))

        is_dup = False
        for idx in candidate_idxs:
            if sig_similarity(sig, kept_sigs[idx]) >= threshold:
                is_dup = True
                if len(dropped_examples) < 5:
                    dropped_examples.append((text, kept[idx][0]))
                break

        if not is_dup:
            new_idx = len(kept)
            kept.append((text, src))
            kept_sigs.append(sig)
            for b, band_val in enumerate(bands):
                buckets[b][band_val].append(new_idx)

    return kept, dropped_examples

after_near, near_dup_examples = lsh_near_dedup(after_exact, threshold=0.5)
print(f"After near dedup    : {len(after_near)}  "
      f"(dropped {len(after_exact) - len(after_near)} near-duplicate paragraphs)")
if near_dup_examples:
    print("  example near-dup pair caught:")
    a, b = near_dup_examples[0]
    print(f"    kept   : {b[:90]!r}")
    print(f"    dropped: {a[:90]!r}")
print()

# ---------- quality filters (from step 8, plus a real C4-paper heuristic:
# reject anything containing the unicode replacement character U+FFFD,
# which C4's actual pipeline does too, as a sign of encoding corruption) ----------
STOPWORDS = {
    "the","be","to","of","and","a","in","that","have","i","it","for","not","on","with",
    "he","as","you","do","at","this","but","his","by","from","they","we","say","her",
    "she","or","an","will","my","one","all","would","there","their","what","so","up",
    "out","if","about","who","get","which","go","me","when","make","can","like","no",
    "just","him","know","take","into","your","some","could","them","see","other","than",
    "then","now","how","its","our","was","is","having",
}

def quality_ok(text):
    if "�" in text:
        return False, "contains unicode replacement char (encoding corruption)"
    words = text.split()
    if len(words) < 8:
        return False, f"too short ({len(words)} words)"
    mwl = sum(len(w) for w in words) / len(words)
    if mwl < 3 or mwl > 10:
        return False, f"mean word length {mwl:.1f} out of [3,10]"
    symbols = len(re.findall(r"[<>{}\[\]/#@$%^&*_=~]", text))
    if symbols / len(words) > 0.15:
        return False, "too symbol-heavy"
    stop_hits = sum(1 for w in re.findall(r"[a-zA-Z']+", text.lower()) if w in STOPWORDS)
    if stop_hits < 2:
        return False, f"too few stopwords ({stop_hits})"
    return True, None

kept_final = []
dropped_final = []
for text, src in after_near:
    ok, reason = quality_ok(text)
    if ok:
        kept_final.append((text, src))
    else:
        dropped_final.append((text, src, reason))

print(f"After quality filter: {len(kept_final)}  "
      f"(dropped {len(dropped_final)} paragraphs)")
print()
print("Sample of what quality filtering dropped (chapter headers, illustration")
print("captions, title-page fragments -- none of these are really 'prose'):")
import random
random.seed(0)
for text, src, reason in random.sample(dropped_final, min(8, len(dropped_final))):
    print(f"  [{src}] {text[:60]!r}  <- {reason}")
print()

# ---------- save results for step 12 ----------
raw_path = os.path.join(DATA_DIR, "corpus_raw.txt")
clean_path = os.path.join(DATA_DIR, "corpus_clean.txt")
with open(raw_path, "w", encoding="utf-8") as f:
    f.write("\n\n".join(t for t, _ in documents))
with open(clean_path, "w", encoding="utf-8") as f:
    f.write("\n\n".join(t for t, _ in kept_final))

print(f"Saved raw corpus   -> {raw_path} ({sum(len(t) for t,_ in documents):,} chars)")
print(f"Saved clean corpus -> {clean_path} ({sum(len(t) for t,_ in kept_final):,} chars)")
print()
print(f"Pipeline summary: {len(documents)} -> {len(after_exact)} (exact dedup) "
      f"-> {len(after_near)} (near dedup) -> {len(kept_final)} (quality filter)")
print(f"Overall: kept {len(kept_final)/len(documents):.1%} of raw paragraphs.")
