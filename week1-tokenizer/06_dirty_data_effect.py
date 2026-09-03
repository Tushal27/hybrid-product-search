"""
STEP 6: The algorithm from steps 3-5 is correct. Now let's prove that a
CORRECT algorithm trained on BAD data still produces a BAD tokenizer.

Raw web scrapes are full of junk that has nothing to do with language:
cookie-consent banners, "buy now!!" spam repeated across thousands of
pages, HTML/ad markup, boilerplate nav text. BPE just counts frequent
byte-pairs -- it has NO concept of "this is junk." If junk repeats more
often than real prose, junk WINS the merge race and eats vocab slots that
should have gone to genuine words.

This is the tokenizer-specific version of "garbage in, garbage out."
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import importlib.util

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("bpe_train", os.path.join(_here, "03_bpe_train.py"))
bpe_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bpe_train)

GOOD_TEXT = bpe_train.CORPUS  # the literary excerpts from step 3

# Realistic junk you'd actually find in a raw Common Crawl-style scrape:
COOKIE_BANNER = " We use cookies to improve your experience. Accept all cookies to continue. "
AD_MARKUP = " <div class='ad-slot'><span>Sponsored</span></div> "
SPAM_VARIANTS = [
    " Buy now and save big today! Limited time offer, buy now! ",
    " Buy now and save big today! Limited time offer, act now! ",
    " Buy now and save more today! Limited time offer, buy now! ",
]

# Simulate a scrape of ~20 web pages: each repeats the cookie banner and an
# ad slot (because it's site-wide boilerplate on every page), a spam blurb
# gets sprinkled in a few times, and the ACTUAL article text (our literary
# corpus) appears only once per "page" worth of junk around it.
pages = []
for i in range(20):
    page = COOKIE_BANNER + AD_MARKUP
    if i % 2 == 0:
        page += SPAM_VARIANTS[i % 3]
    pages.append(page)
# the real content, diluted among 20 pages of junk
DIRTY_CORPUS = "".join(pages) + GOOD_TEXT

print(f"Junk characters   : {sum(len(p) for p in pages):>6}")
print(f"Real prose chars  : {len(GOOD_TEXT):>6}")
print(f"Junk is {sum(len(p) for p in pages) / len(GOOD_TEXT):.1f}x the volume of real content")
print("(this ratio is realistic -- boilerplate often outweighs unique content on the open web)")
print()

merges, vocab = bpe_train.train(DIRTY_CORPUS, num_merges=30, verbose=True)

print()
junk_markers = ["cookie", "Accept", "Sponsored", "ad-slot", "Buy now", "save", "Limited", "offer", "div", "span"]


def looks_like_junk(token_bytes: bytes) -> bool:
    text = token_bytes.decode("utf-8", errors="ignore")
    return any(marker.lower() in text.lower() for marker in junk_markers)


junk_merge_count = sum(1 for new_id in merges.values() if looks_like_junk(vocab[new_id]))
print(f"Of the 30 merges learned, {junk_merge_count} are junk/boilerplate fragments,")
print(f"not general-purpose English. That's {junk_merge_count/30:.0%} of our tiny vocab")
print("budget spent on 'cookie', 'Sponsored', 'Buy now' instead of real language --")
print("and this is with junk only ~2x the real content. Real Common Crawl dumps can")
print("be >90% boilerplate/spam/low-value text before cleaning. At GPT-4 scale (100k")
print("merges over trillions of tokens) that means potentially TENS of THOUSANDS of")
print("vocab slots wasted the same way, unless the data is cleaned first.")
