"""
Side-by-side: OUR tiny hand-trained tokenizer (300 merges, trained on a few
paragraphs of 19th-century novels) vs the REAL GPT-4 tokenizer (cl100k_base,
100256 merges, trained on trillions of tokens of diverse web/code/book text).

Edit `text` below and rerun to compare on your own sentences. Try:
 - a sentence close to the training novels (should compress well on OURS)
 - modern slang, code, or a totally different topic (OURS should do badly,
   GPT-4 should still do fine -- that gap IS the lesson)
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import importlib.util


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_here = os.path.dirname(os.path.abspath(__file__))
bpe_train = load("bpe_train", os.path.join(_here, "03_bpe_train.py"))
bpe_codec = load("bpe_codec", os.path.join(_here, "04_encode_decode.py"))
verify = load("verify", os.path.join(_here, "05_verify_gpt4.py"))  # also runs its own 8 test cases once

print("Training OUR tiny tokenizer (300 merges on ~1KB of 19th-century prose)...")
OUR_MERGES, OUR_VOCAB = bpe_train.train(bpe_train.CORPUS, num_merges=300, verbose=False)
print(f"Our vocab size: {len(OUR_VOCAB)}")

print("Loading REAL GPT-4 tokenizer (cl100k_base, 100256 tokens)...")
gpt4 = verify.OurGPT4Tokenizer()
print(f"GPT-4 vocab size: {len(gpt4.vocab)}")
print()


def compare(text: str):
    our_ids = bpe_codec.encode(text, OUR_MERGES)
    our_pieces = [bpe_codec.decode([i], OUR_VOCAB) for i in our_ids]

    gpt4_ids = gpt4.encode(text)
    gpt4_pieces = [gpt4.decode([i]) for i in gpt4_ids]

    raw_bytes = len(text.encode("utf-8"))

    print(f"text: {text!r}")
    print(f"  raw bytes        : {raw_bytes}")
    print(f"  OURS  ({len(OUR_VOCAB):>6}-token vocab): {len(our_ids):>3} tokens, "
          f"{raw_bytes/len(our_ids):.2f}x compression")
    print(f"          pieces: {our_pieces}")
    print(f"  GPT-4 ({len(gpt4.vocab):>6}-token vocab): {len(gpt4_ids):>3} tokens, "
          f"{raw_bytes/len(gpt4_ids):.2f}x compression")
    print(f"          pieces: {gpt4_pieces}")
    print()


# ---- try your own text below ----
test_sentences = [
    "It was the age of wisdom, it was the age of foolishness.",   # near training data
    "I need to deploy the Kubernetes cluster before the sprint review.",  # modern/technical
    "wanna grab boba after class lol",                             # modern slang
]
for text in test_sentences:
    compare(text)

print("=" * 70)
print("Notice: on text CLOSE to what ours was trained on, the compression gap")
print("shrinks a lot. On modern/technical/slang text -- things that simply")
print("didn't exist in a handful of 19th-century paragraphs -- ours falls back")
print("to near byte-level (barely any merges apply) while GPT-4, trained on a")
print("vastly larger and more diverse corpus, still compresses well. Same")
print("algorithm both times. The only difference is what each one was trained")
print("on -- vocab size AND data diversity both matter.")
