"""
STEP 1: What an embedding actually IS, mechanically.

You already know a tokenizer turns text into a list of integer ids
(e.g. "Hello" -> [9906]). An embedding layer's entire job is absurdly
simple: it's a big matrix of shape (vocab_size, embedding_dim), and
"embedding a token" just means READING ONE ROW of that matrix by its
token id. That's it. No math happens here beyond an array index.

  embedding_matrix[9906]  ==  the embedding vector for token 9906

The "learning" in an LLM doesn't happen in some separate embedding
algorithm -- this matrix is just one more set of weights that gets
updated by ordinary backprop, exactly like every other weight in the
network. Nothing about embeddings is special mechanically; what's
special is what ends up STORED in each row after training.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import importlib.util
import numpy as np

np.random.seed(0)

# reuse the real GPT-4 tokenizer you already built and verified
_tok_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "week1-tokenizer")
_spec = importlib.util.spec_from_file_location("verify", os.path.join(_tok_dir, "05_verify_gpt4.py"))
verify = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verify)
tok = verify.OurGPT4Tokenizer()

VOCAB_SIZE = 100_277          # cl100k_base's actual vocab size
EMBED_DIM = 8                 # tiny on purpose so we can print whole vectors;
                               # real models use 768-12288+ dimensions

# THIS is the entire "embedding layer": one matrix, randomly initialized.
# (real training starts from random too -- it's the training that gives
# the rows meaning, not the initialization.)
embedding_matrix = np.random.randn(VOCAB_SIZE, EMBED_DIM) * 0.02

text = "Hello cat dog"
ids = tok.encode(text)
pieces = [tok.decode([i]) for i in ids]

print(f"text : {text!r}")
print(f"ids  : {ids}")
print(f"pieces: {pieces}")
print()

print("'Embedding' each token = indexing one row out of the matrix:")
for tid, piece in zip(ids, pieces):
    vec = embedding_matrix[tid]         # <-- this line IS the embedding lookup
    print(f"  id {tid:>6} ({piece!r:>10}) -> {np.round(vec, 3)}")
print()

print(f"embedding_matrix.shape = {embedding_matrix.shape}")
print(f"That's {VOCAB_SIZE:,} x {EMBED_DIM} = {VOCAB_SIZE*EMBED_DIM:,} numbers total,")
print("all of which start as random noise and get nudged by gradient descent")
print("during training until nearby-meaning tokens end up with similar rows.")
print()
print("Right now, nothing is learned yet -- 'cat' and 'dog' have RANDOM vectors")
print("with no relationship to each other, even though they're semantically")
print("related words. Next script: prove that quantitatively.")
