"""
STEP 9: Close the loop. Run the FULL pipeline (exact dedup -> near dedup ->
quality filter, from steps 7-8) on the same dirty scrape from step 6, then
retrain BPE (step 3's algorithm, completely unchanged) on both versions and
compare the merges side by side.

This is the whole argument in one script: same algorithm, different data
in -> measurably different vocab out.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import re
import hashlib
import importlib.util

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("bpe_train", os.path.join(_here, "03_bpe_train.py"))
bpe_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bpe_train)

GOOD_TEXT = bpe_train.CORPUS
GOOD_PARAGRAPHS = [p.strip() for p in GOOD_TEXT.strip().split("\n\n") if p.strip()]

COOKIE_BANNER = "We use cookies to improve your experience. Accept all cookies to continue."
AD_MARKUP = "<div class='ad-slot'><span>Sponsored</span></div>"
SPAM_VARIANTS = [
    "Buy now and save big today! Limited time offer, buy now!",
    "Buy now and save big today! Limited time offer, act now!",
    "Buy now and save more today! Limited time offer, buy now!",
]

# same shape of "scrape" as step 6/7: lots of repeated boilerplate + spam,
# real content much rarer.
documents = (
    [COOKIE_BANNER] * 8
    + [AD_MARKUP] * 8
    + SPAM_VARIANTS * 3
    + GOOD_PARAGRAPHS
)

# ---------- cleaning pipeline (steps 7 + 8, condensed) ----------
def normalize(text): return " ".join(text.split()).lower()

def exact_dedup(docs):
    seen, kept = set(), []
    for d in docs:
        h = hashlib.sha256(normalize(d).encode()).hexdigest()
        if h not in seen:
            seen.add(h); kept.append(d)
    return kept

def shingles(text, k=3):
    words = normalize(text).split()
    if len(words) < k:
        return {tuple(words)}
    return {tuple(words[i:i+k]) for i in range(len(words) - k + 1)}

def minhash_sig(shingle_set, num_hashes=64):
    return [min((int(hashlib.md5(f"{s_}:{s}".encode()).hexdigest(), 16) for s in shingle_set), default=0)
            for s_ in range(num_hashes)]

def sig_sim(a, b): return sum(1 for x, y in zip(a, b) if x == y) / len(a)

def near_dedup(docs, threshold=0.5):
    kept, sigs = [], []
    for d in docs:
        sig = minhash_sig(shingles(d))
        if not any(sig_sim(sig, s) >= threshold for s in sigs):
            kept.append(d); sigs.append(sig)
    return kept

STOPWORDS = {
    "the","be","to","of","and","a","in","that","have","i","it","for","not","on","with",
    "he","as","you","do","at","this","but","his","by","from","they","we","say","her",
    "she","or","an","will","my","one","all","would","there","their","what","so","up",
    "out","if","about","who","get","which","go","me","when","make","can","like","no",
    "just","him","know","take","into","your","some","could","them","see","other","than",
    "then","now","how","its","our","was","is","having",
}
BOILERPLATE_PHRASES = ["cookies", "accept all", "sponsored", "terms of service"]

def quality_ok(text):
    words = text.split()
    if len(words) < 8:
        return False
    mwl = sum(len(w) for w in words) / len(words)
    if mwl < 3 or mwl > 10:
        return False
    symbols = len(re.findall(r"[<>{}\[\]/#@$%^&*_=~]", text))
    if symbols / len(words) > 0.15:
        return False
    stop_hits = sum(1 for w in re.findall(r"[a-zA-Z']+", text.lower()) if w in STOPWORDS)
    if stop_hits < 2:
        return False
    if any(p in text.lower() for p in BOILERPLATE_PHRASES):
        return False
    return True

cleaned = [d for d in near_dedup(exact_dedup(documents)) if quality_ok(d)]

DIRTY_CORPUS = " ".join(documents)
CLEAN_CORPUS = " ".join(cleaned)

print(f"Raw documents        : {len(documents)}")
print(f"After full cleaning  : {len(cleaned)}  -> {cleaned}")
print()
print(f"Dirty corpus length  : {len(DIRTY_CORPUS)} chars")
print(f"Clean corpus length  : {len(CLEAN_CORPUS)} chars")
print()

junk_markers = ["cookie", "Accept", "Sponsored", "ad-slot", "Buy now", "save", "Limited", "offer", "div", "span"]
def looks_like_junk(b):
    t = b.decode("utf-8", errors="ignore")
    return any(m.lower() in t.lower() for m in junk_markers)

N_MERGES = 30
print("=" * 60)
print(f"Training on DIRTY corpus ({N_MERGES} merges):")
dirty_merges, dirty_vocab = bpe_train.train(DIRTY_CORPUS, num_merges=N_MERGES, verbose=False)
dirty_junk = sum(1 for nid in dirty_merges.values() if looks_like_junk(dirty_vocab[nid]))
print([dirty_vocab[nid] for nid in dirty_merges.values()])
print(f"-> {dirty_junk}/{N_MERGES} merges are junk fragments ({dirty_junk/N_MERGES:.0%})")
print()

print("=" * 60)
print(f"Training on CLEAN corpus ({N_MERGES} merges):")
clean_merges, clean_vocab = bpe_train.train(CLEAN_CORPUS, num_merges=N_MERGES, verbose=False)
clean_junk = sum(1 for nid in clean_merges.values() if looks_like_junk(clean_vocab[nid]))
print([clean_vocab[nid] for nid in clean_merges.values()])
print(f"-> {clean_junk}/{N_MERGES} merges are junk fragments ({clean_junk/N_MERGES:.0%})")
print()

print("=" * 60)
print(f"SAME algorithm (steps 3-5), SAME merge count. Junk fraction went from")
print(f"{dirty_junk/N_MERGES:.0%} (dirty) to {clean_junk/N_MERGES:.0%} (clean) purely from cleaning the input data.")
print("This is the entire 'why training data matters' argument, made concrete")
print("instead of asserted: the algorithm was never the variable. The data was.")
