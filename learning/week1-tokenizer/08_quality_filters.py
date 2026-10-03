"""
STEP 8: Quality heuristic filters.

Dedup (step 7) only removes documents that are copies of EACH OTHER. It has
no opinion about whether a unique document is any good. Our cookie banner
and ad markup are each unique after dedup -- dedup correctly leaves them
alone. Quality filters are the separate, independent pass that looks at
EACH document on its own and asks "does this look like real language?"

These are simplified versions of heuristics used in real pipelines (the
Gopher paper, C4/T5, RedPajama, Dolma all publish variants of these). I'm
not reproducing anyone's exact published thresholds -- these are
reasonable approximations for learning the SHAPE of the technique. In a
real pipeline you'd tune thresholds against your own data + human review.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import re

DOCS = [
    "We use cookies to improve your experience. Accept all cookies to continue.",
    "<div class='ad-slot'><span>Sponsored</span></div>",
    "Buy now and save big today! Limited time offer, buy now!",
    "It was the best of times, it was the worst of times, it was the age of "
    "wisdom, it was the age of foolishness.",
    "It is a truth universally acknowledged, that a single man in possession "
    "of a good fortune, must be in want of a wife.",
    "Call me Ishmael. Some years ago, never mind how long precisely, having "
    "little or no money in my purse.",
]

# a broader top-N list of very common English function words -- Gopher's
# actual rule requires the document to contain at least a couple of these,
# on the theory that real prose always uses them heavily but keyword-stuffed
# spam / boilerplate / markup often doesn't. (A first pass at this with only
# ~13 words wrongly dropped a genuine Melville quote for using "some", "how",
# "or", "no", "my" instead of "the"/"and"/"of" -- too small a list makes the
# filter trigger-happy against perfectly good prose. This is the realistic
# size such lists need to be before they stop producing false positives.)
STOPWORDS = {
    "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for",
    "not", "on", "with", "he", "as", "you", "do", "at", "this", "but", "his",
    "by", "from", "they", "we", "say", "her", "she", "or", "an", "will", "my",
    "one", "all", "would", "there", "their", "what", "so", "up", "out", "if",
    "about", "who", "get", "which", "go", "me", "when", "make", "can", "like",
    "no", "just", "him", "know", "take", "into", "your", "some", "could",
    "them", "see", "other", "than", "then", "now", "how", "its", "our", "was",
    "is", "having",
}

# some real production pipelines (C4's paper is explicit about this) also
# keep a small blocklist of specific boilerplate PHRASES -- things like
# "terms of service", "lorem ipsum", "all rights reserved" -- because
# boilerplate like a cookie banner is grammatically perfect English (it
# genuinely uses "we", "to", "your" the way real prose does), so no purely
# grammatical heuristic can separate it from real content. Only knowing
# the SPECIFIC phrase is boilerplate can catch it here; at scale, exact/
# near dedup (step 7) is what actually removes most boilerplate, since it
# repeats verbatim across thousands of pages -- this phrase list is a
# supplementary catch for cases where dedup hasn't run yet or missed one.
BOILERPLATE_PHRASES = ["cookies", "accept all", "sponsored", "terms of service", "all rights reserved"]


def word_count(text: str) -> int:
    return len(text.split())


def mean_word_length(text: str) -> float:
    words = text.split()
    return sum(len(w) for w in words) / len(words) if words else 0.0


def symbol_to_word_ratio(text: str) -> float:
    symbols = len(re.findall(r"[<>{}\[\]/#@$%^&*_=~]", text))
    words = word_count(text)
    return symbols / words if words else float("inf")


def stopword_count(text: str) -> int:
    words = re.findall(r"[a-zA-Z']+", text.lower())
    return sum(1 for w in words if w in STOPWORDS)


def alpha_word_fraction(text: str) -> float:
    """fraction of words that are mostly alphabetic (catches HTML/markup
    'words' like '<div' or 'class='ad-slot'' that are mostly symbols)."""
    words = text.split()
    if not words:
        return 0.0
    alpha_words = sum(1 for w in words if sum(c.isalpha() for c in w) / len(w) > 0.6)
    return alpha_words / len(words)


def quality_report(text: str) -> dict:
    return {
        "words": word_count(text),
        "mean_word_len": round(mean_word_length(text), 2),
        "symbol_ratio": round(symbol_to_word_ratio(text), 2),
        "stopwords": stopword_count(text),
        "alpha_frac": round(alpha_word_fraction(text), 2),
    }


def passes_quality_filter(text: str) -> tuple[bool, list[str]]:
    reasons = []
    r = quality_report(text)
    if r["words"] < 8:
        reasons.append(f"too short ({r['words']} words)")
    if r["mean_word_len"] < 3 or r["mean_word_len"] > 10:
        reasons.append(f"mean word length {r['mean_word_len']} out of [3,10]")
    if r["symbol_ratio"] > 0.15:
        reasons.append(f"too symbol-heavy ({r['symbol_ratio']} symbols/word)")
    if r["stopwords"] < 2:
        reasons.append(f"too few stopwords ({r['stopwords']}) -- doesn't read as real prose")
    if r["alpha_frac"] < 0.8:
        reasons.append(f"only {r['alpha_frac']:.0%} of words are mostly-alphabetic")
    lower = text.lower()
    hit = next((p for p in BOILERPLATE_PHRASES if p in lower), None)
    if hit:
        reasons.append(f"matches known boilerplate phrase {hit!r}")
    return (len(reasons) == 0, reasons)


kept, dropped = [], []
for doc in DOCS:
    ok, reasons = passes_quality_filter(doc)
    report = quality_report(doc)
    print(f"{'KEEP' if ok else 'DROP'}: {doc[:55]!r}")
    print(f"      {report}")
    if not ok:
        print(f"      reasons: {reasons}")
        dropped.append(doc)
    else:
        kept.append(doc)
    print()

print(f"Kept {len(kept)}/{len(DOCS)} documents.")
print()
print("The cookie banner and ad markup get dropped even though dedup already")
print("said they were 'unique' -- dedup and quality filtering are answering two")
print("completely different questions ('have I seen this before?' vs 'is this")
print("any good on its own?') and production pipelines need BOTH passes.")
