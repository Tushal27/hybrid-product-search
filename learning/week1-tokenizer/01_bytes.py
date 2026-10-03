"""
STEP 1: Why does GPT-4's tokenizer start from BYTES, not characters?

Naive idea: vocab = every unique character in the training data.
Problem: Unicode has 150,000+ possible characters (emoji, CJK, accents...).
If you build vocab from "characters seen in training data", you WILL hit a
character at inference time that wasn't in your vocab -> unknown token (<unk>)
-> information is silently lost. Bad for a production model.

GPT-2/GPT-4's fix: don't operate on characters. Operate on raw UTF-8 BYTES.
A byte only has 256 possible values (0-255). ANY string in ANY language,
emoji, or even raw binary can be represented as a sequence of bytes.
So byte-level vocab has a hard guarantee: 100% coverage, zero <unk> tokens,
ever. BPE merges then run on TOP of these bytes to build bigger chunks.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

examples = [
    "hi",
    "café",
    "日本語",
    "🚀",
    "naïve",
]

for text in examples:
    b = text.encode("utf-8")
    print(f"text: {text!r}")
    print(f"  chars           : {len(text)}  -> {list(text)}")
    print(f"  utf-8 bytes     : {len(b)}  -> {list(b)}")
    print(f"  bytes as tokens : {[bytes([x]) for x in b]}")
    print()

print("Notice: 'hi' -> 2 chars -> 2 bytes (plain ASCII, 1 byte each).")
print("But 'café', '日本語', '🚀' need MULTIPLE bytes per character.")
print("A character-level vocab would need a slot for every one of these.")
print("A byte-level vocab only ever needs 256 slots (ids 0-255) as the base.")
print()
print("This is exactly the '101 502 2148' style small integers you saw for")
print("tokens in your roadmap image -- those start life as byte ids 0-255,")
print("then BPE merges combine common byte sequences into single new ids")
print("(e.g. the bytes for a whole common word can become ONE token id).")
