"""
STEP 7: Deduplication. Two different problems, two different techniques.

1. EXACT duplicates (byte-identical documents, e.g. the same cookie-banner
   boilerplate on every page of a site) -> hash the whole document, drop
   repeats. Cheap, exact, catches nothing that differs by even one char.

2. NEAR duplicates (same spam blurb reworded slightly, e.g. "Buy now and
   save big today! ... buy now!" vs "...act now!") -> exact hashing won't
   catch these since the bytes differ. Need similarity, not equality.
   Real pipelines (CCNet, RedPajama, Dolma, GPT-3's own dataset paper) use
   MinHash + LSH for this at billion-document scale, because comparing
   every document to every other document (O(n^2)) is impossible at that
   size. We implement real MinHash below (just skip the LSH indexing
   trick, since our corpus is small enough for the O(n^2) comparison to
   be fine for learning purposes -- the LSH part is a *speed* optimization,
   not a correctness one).
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import hashlib

COOKIE_BANNER = "We use cookies to improve your experience. Accept all cookies to continue."
AD_MARKUP = "<div class='ad-slot'><span>Sponsored</span></div>"
SPAM_VARIANTS = [
    "Buy now and save big today! Limited time offer, buy now!",
    "Buy now and save big today! Limited time offer, act now!",
    "Buy now and save more today! Limited time offer, buy now!",
]
REAL_PARAGRAPHS = [
    "It was the best of times, it was the worst of times, it was the age of "
    "wisdom, it was the age of foolishness.",
    "It is a truth universally acknowledged, that a single man in possession "
    "of a good fortune, must be in want of a wife.",
    "Call me Ishmael. Some years ago, never mind how long precisely, having "
    "little or no money in my purse.",
]

# Build a document set like a real scrape: boilerplate repeated exactly on
# every page, spam reworded slightly each time, real content unique.
documents = (
    [COOKIE_BANNER] * 8
    + [AD_MARKUP] * 8
    + SPAM_VARIANTS * 3          # each variant appears 3x with the SAME wording
    + REAL_PARAGRAPHS
)
print(f"Raw document count: {len(documents)}")
print()

# ---------- 1. EXACT dedup ----------
def normalize(text: str) -> str:
    return " ".join(text.split()).lower()


def exact_dedup(docs: list[str]) -> list[str]:
    seen_hashes = set()
    kept = []
    for doc in docs:
        h = hashlib.sha256(normalize(doc).encode("utf-8")).hexdigest()
        if h not in seen_hashes:
            seen_hashes.add(h)
            kept.append(doc)
    return kept


after_exact = exact_dedup(documents)
print(f"After EXACT dedup  : {len(after_exact)} documents  "
      f"(dropped {len(documents) - len(after_exact)} byte-identical repeats)")
for d in after_exact:
    print(f"  - {d[:60]!r}")
print()
print("Notice the 3 spam variants all SURVIVED exact dedup -- they're not")
print("byte-identical, so hashing the whole string treats them as 3 different")
print("documents. This is exactly the gap near-dedup exists to close.")
print()


# ---------- 2. NEAR dedup: shingling + Jaccard + MinHash ----------
def shingles(text: str, k: int = 3) -> set[tuple[str, ...]]:
    """k-gram of words. Two texts that share most of their k-word runs are
    near-duplicates even if individual words differ elsewhere."""
    words = normalize(text).split()
    if len(words) < k:
        return {tuple(words)}
    return {tuple(words[i:i + k]) for i in range(len(words) - k + 1)}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


NUM_HASHES = 64


def minhash_signature(shingle_set: set[tuple[str, ...]]) -> list[int]:
    """Approximate a document's shingle set as NUM_HASHES small integers.
    Two documents with similar shingle sets will have similar signatures
    -- the fraction of matching signature slots approximates their true
    Jaccard similarity, without ever storing the full shingle sets."""
    sig = []
    for seed in range(NUM_HASHES):
        min_val = min(
            (int(hashlib.md5(f"{seed}:{s}".encode()).hexdigest(), 16) for s in shingle_set),
            default=0,
        )
        sig.append(min_val)
    return sig


def signature_similarity(sig_a: list[int], sig_b: list[int]) -> float:
    return sum(1 for a, b in zip(sig_a, sig_b) if a == b) / len(sig_a)


print("Pairwise similarity of the 3 (reworded) spam variants:")
spam_shingle_sets = [shingles(s) for s in SPAM_VARIANTS]
spam_sigs = [minhash_signature(s) for s in spam_shingle_sets]
for i in range(len(SPAM_VARIANTS)):
    for j in range(i + 1, len(SPAM_VARIANTS)):
        true_jaccard = jaccard(spam_shingle_sets[i], spam_shingle_sets[j])
        approx = signature_similarity(spam_sigs[i], spam_sigs[j])
        print(f"  variant {i} vs {j}: true Jaccard={true_jaccard:.2f}   "
              f"MinHash estimate={approx:.2f}")

print()
print("Compare against a spam variant vs a REAL paragraph (should be near 0):")
real_shingles = shingles(REAL_PARAGRAPHS[0])
print(f"  spam[0] vs real[0]: true Jaccard="
      f"{jaccard(spam_shingle_sets[0], real_shingles):.2f}")
print()


def near_dedup(docs: list[str], threshold: float = 0.5) -> list[str]:
    kept: list[str] = []
    kept_sigs: list[list[int]] = []
    for doc in docs:
        sig = minhash_signature(shingles(doc))
        is_dup = any(signature_similarity(sig, k) >= threshold for k in kept_sigs)
        if not is_dup:
            kept.append(doc)
            kept_sigs.append(sig)
    return kept


after_near = near_dedup(after_exact, threshold=0.5)
print(f"After NEAR dedup   : {len(after_near)} documents "
      f"(started from the {len(after_exact)} that survived exact dedup)")
for d in after_near:
    print(f"  - {d[:60]!r}")
print()
print(f"Final: {len(documents)} raw -> {len(after_near)} after both dedup passes.")
print("Real production pipelines run this same two-stage approach (exact")
print("hash dedup, then MinHash+LSH near-dedup) across billions of documents;")
print("the LSH part just makes step 2 sub-quadratic instead of comparing")
print("every document to every other document like we did here.")
