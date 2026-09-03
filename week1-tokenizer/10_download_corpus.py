"""
STEP 10: Download a REAL dataset instead of hand-written toy paragraphs.

6 public-domain novels from Project Gutenberg (~4MB of real text). This is
still tiny compared to a real pretraining corpus (trillions of tokens), but
it's genuinely messy in the same WAYS real data is messy: every single file
has Project Gutenberg's license boilerplate repeated verbatim at the start
and end, plus tables of contents, chapter headers, and illustration
captions that aren't really "prose" -- all real analogs of the synthetic
junk we made up in steps 6-9.

Downloads are cached to disk so this only hits the network once.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import urllib.request

BOOKS = {
    1342: "pride_and_prejudice",
    84: "frankenstein",
    11: "alice_in_wonderland",
    1661: "sherlock_holmes",
    98: "tale_of_two_cities",
    2701: "moby_dick",
}

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def download_book(book_id: int, name: str) -> str:
    path = os.path.join(DATA_DIR, f"{name}.txt")
    if os.path.exists(path):
        print(f"  {name}: already cached")
        return path
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    with open(path, "wb") as f:
        f.write(data)
    print(f"  {name}: downloaded {len(data):,} bytes")
    return path


if __name__ == "__main__":
    print(f"Downloading to {DATA_DIR}")
    paths = {}
    for book_id, name in BOOKS.items():
        paths[name] = download_book(book_id, name)

    total_bytes = sum(os.path.getsize(p) for p in paths.values())
    print()
    print(f"Total corpus size: {total_bytes:,} bytes (~{total_bytes/1e6:.1f} MB)")
    print()
    # peek at the start of one file to see the real boilerplate we'll need to clean
    with open(paths["pride_and_prejudice"], encoding="utf-8") as f:
        preview = f.read(600)
    print("Preview of pride_and_prejudice.txt (notice the Gutenberg boilerplate):")
    print("-" * 60)
    print(preview)
    print("-" * 60)
