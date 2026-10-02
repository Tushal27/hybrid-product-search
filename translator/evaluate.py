"""
Score the translator on 300 held-out OPUS-100 sentence pairs it never trained on.
BLEU (higher = closer to the human French) plus a side-by-side of the first few.

Usage:
    python translator/evaluate.py base      # the untouched Qwen with the translate prompt
    python translator/evaluate.py tuned     # base + our LoRA patch
"""

import sys
import time
import sacrebleu
from datasets import load_dataset
from common import load_base, load_translator, translate

mode = sys.argv[1] if len(sys.argv) > 1 else "tuned"
N = 300
val = load_dataset("Helsinki-NLP/opus-100", "en-fr", split="test")
pairs = [(e["translation"]["en"].strip(), e["translation"]["fr"].strip()) for e in val
         if 3 <= len(e["translation"]["en"]) <= 160][:N]
english = [p[0] for p in pairs]
reference = [p[1] for p in pairs]

tokenizer, model = load_base() if mode == "base" else load_translator()
t = time.time()
hypo = translate(tokenizer, model, english)
bleu = sacrebleu.corpus_bleu(hypo, [reference])
print(f"\n[{mode}] BLEU on {len(pairs)} unseen sentences: {bleu.score:.1f}   ({time.time() - t:.0f}s)\n")
for en, fr, ref in list(zip(english, hypo, reference))[:8]:
    print(f"EN : {en}\nOUT: {fr}\nREF: {ref}\n")
