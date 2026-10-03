"""
WEEK 2 CAPSTONE: One prompt, in your own words, all the way to a
response -- with EVERY stage printed, using YOUR tokenizer (trained live
on the real corpus from week1-tokenizer) feeding YOUR transformer stack
(week2-inference 01-05). Nothing here is a toy vocab or a canned example
anymore -- this is the actual end-to-end machine.

Run it with your own prompt:
    .venv\\Scripts\\python.exe week2-inference\\06_prompt_to_response_capstone.py "your prompt here"
With no argument, it asks you for one; with no stdin available either, it
falls back to a default so the script still runs non-interactively.

THE ONE HONEST CAVEAT, up front: the transformer's weights below are
RANDOM. Nothing has been trained yet (that's deliberately outside this
5-week roadmap's scope -- see week3-memory-performance and beyond, which
assume PRETRAINED weights). So the generated "response" will be real,
grammatical-looking word-pieces from YOUR tokenizer's vocabulary, strung
together in a meaningless order -- not coherent English. That's expected,
and it's not the point. The point is watching every single gear turn,
with zero magic hidden anywhere:

  raw text
    -> UTF-8 bytes -> regex pre-split chunks -> BPE token ids   (week 1)
    -> token embedding lookup + positional encoding             (week 2, step 1)
    -> N stacked causal transformer blocks                      (week 2, step 1)
    -> output head -> logits -> softmax                         (week 2, step 2)
    -> repetition penalties -> temperature -> top-p -> sample    (week 2, steps 3-4)
    -> decode the sampled id back to actual text                (week 1)
    -> append, repeat
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import time
import importlib.util
import numpy as np

np.random.seed(0)
rng = np.random.default_rng(0)


# ------------------------- reuse the real tokenizer from week 1 -------------------------
_tok_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "week1-tokenizer")


def _load(module_name, filename):
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(_tok_dir, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bpe_train = _load("bpe_train", "03_bpe_train.py")
fast_bpe = _load("fast_bpe", "13_fast_bpe_train.py")
enc_dec = _load("enc_dec", "04_encode_decode.py")

encode = enc_dec.encode
decode = enc_dec.decode

# ---------------------------------- step A: train the tokenizer ----------------------------------
NUM_MERGES = 1000

print("=" * 78)
print("STEP A -- train YOUR byte-level BPE tokenizer on the real corpus")
print("=" * 78)
corpus_path = os.path.join(_tok_dir, "data", "corpus_clean.txt")
with open(corpus_path, encoding="utf-8") as f:
    corpus_text = f.read()

t0 = time.time()
merges, vocab = fast_bpe.train_fast(corpus_text, num_merges=NUM_MERGES, verbose=False)
train_time = time.time() - t0
VOCAB_SIZE = len(vocab)
print(f"Trained on {len(corpus_text):,} characters of real text "
      f"(Alice in Wonderland, Moby Dick, Frankenstein, ...)")
print(f"{NUM_MERGES} merges learned in {train_time:.2f}s -> vocab size = {VOCAB_SIZE} "
      f"(256 raw bytes + {NUM_MERGES} learned merges)")
print()


# ---------------------------------- step B: get a prompt ----------------------------------
DEFAULT_PROMPT = "It was the age of wisdom"

if len(sys.argv) > 1:
    prompt = " ".join(sys.argv[1:])
else:
    try:
        prompt = input("Enter a prompt (Enter for default): ").strip()
    except EOFError:
        prompt = ""
    if not prompt:
        prompt = DEFAULT_PROMPT
        print(f"(no prompt given -- using default: {prompt!r})")

print("=" * 78)
print("STEP B -- your prompt, encoded")
print("=" * 78)
print(f"Prompt text        : {prompt!r}")
prompt_bytes = prompt.encode("utf-8")
print(f"UTF-8 bytes         : {list(prompt_bytes)}")
prompt_ids = encode(prompt, merges)
print(f"BPE token ids       : {prompt_ids}")
print(f"num tokens          : {len(prompt_ids)}  (vs {len(prompt_bytes)} raw bytes -> "
      f"{len(prompt_bytes)/len(prompt_ids):.2f}x compression)")
print("Each token, decoded individually (this is literally what the model 'sees'):")
for tid in prompt_ids:
    piece = vocab[tid].decode("utf-8", errors="replace")
    print(f"    id {tid:>5}  ->  {piece!r}")
print(f"Round-trip check: decode(encode(prompt)) == prompt -> {decode(prompt_ids, vocab) == prompt}")
print()


# ---------------------------------- step C: the model (week 2, steps 1-2) ----------------------------------
EMBED_DIM = 64
NUM_HEADS = 8
NUM_LAYERS = 4
FF_HIDDEN_DIM = EMBED_DIM * 4
MAX_SEQ_LEN = 256


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def multi_head_attention(X, Wq, Wk, Wv, Wo, num_heads, mask=None):
    n_tokens, embed_dim = X.shape
    d_k = embed_dim // num_heads
    Q_full, K_full, V_full = X @ Wq, X @ Wk, X @ Wv
    head_outputs = []
    for h in range(num_heads):
        sl = slice(h * d_k, (h + 1) * d_k)
        Q_h, K_h, V_h = Q_full[:, sl], K_full[:, sl], V_full[:, sl]
        scores = Q_h @ K_h.T / np.sqrt(d_k)
        if mask is not None:
            scores = np.where(mask, scores, -np.inf)
        head_outputs.append(softmax(scores) @ V_h)
    return np.concatenate(head_outputs, axis=-1) @ Wo


def layer_norm(x, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def positional_encoding(seq_len, d_model):
    pos = np.arange(seq_len)[:, None]
    i = np.arange(d_model)[None, :]
    angle_rates = 1 / (10000 ** ((2 * (i // 2)) / d_model))
    angles = pos * angle_rates
    pe = np.zeros((seq_len, d_model))
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe


class TransformerBlock:
    def __init__(self, embed_dim, num_heads, ff_hidden_dim, seed):
        block_rng = np.random.default_rng(seed)
        scale = 0.3
        self.Wq = block_rng.standard_normal((embed_dim, embed_dim)) * scale
        self.Wk = block_rng.standard_normal((embed_dim, embed_dim)) * scale
        self.Wv = block_rng.standard_normal((embed_dim, embed_dim)) * scale
        self.Wo = block_rng.standard_normal((embed_dim, embed_dim)) * scale
        self.W1 = block_rng.standard_normal((embed_dim, ff_hidden_dim)) * scale
        self.b1 = np.zeros(ff_hidden_dim)
        self.W2 = block_rng.standard_normal((ff_hidden_dim, embed_dim)) * scale
        self.b2 = np.zeros(embed_dim)
        self.num_heads = num_heads

    def forward(self, x, mask=None):
        x = layer_norm(x + multi_head_attention(x, self.Wq, self.Wk, self.Wv, self.Wo,
                                                  self.num_heads, mask=mask))
        ff_out = gelu(x @ self.W1 + self.b1) @ self.W2 + self.b2
        return layer_norm(x + ff_out)


class MiniGPT:
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_hidden_dim,
                 max_seq_len, seed=0):
        model_rng = np.random.default_rng(seed)
        self.token_embed = model_rng.standard_normal((vocab_size, embed_dim)) * 0.02
        self.pos_encoding = positional_encoding(max_seq_len, embed_dim)
        self.blocks = [TransformerBlock(embed_dim, num_heads, ff_hidden_dim, seed=100 + i)
                        for i in range(num_layers)]
        self.W_head = model_rng.standard_normal((embed_dim, vocab_size)) * 0.3

    def forward(self, token_ids):
        n = len(token_ids)
        x = self.token_embed[token_ids] + self.pos_encoding[:n]
        mask = np.tril(np.ones((n, n), dtype=bool))
        for block in self.blocks:
            x = block.forward(x, mask=mask)
        x = layer_norm(x)
        return x @ self.W_head  # logits, shape (n_tokens, vocab_size)


print("=" * 78)
print("STEP C -- build the model, sized to YOUR real vocabulary")
print("=" * 78)
print(f"vocab_size={VOCAB_SIZE}  embed_dim={EMBED_DIM}  num_heads={NUM_HEADS}  "
      f"num_layers={NUM_LAYERS}  ff_hidden_dim={FF_HIDDEN_DIM}")
print("(weights are RANDOM -- no training has happened. See the module docstring.)")
model = MiniGPT(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_HIDDEN_DIM, MAX_SEQ_LEN)
print()


# ---------------------------------- step D: decoding strategy (week 2, steps 3-4) ----------------------------------
TEMPERATURE = 0.8
TOP_P = 0.9
FREQ_PENALTY = 0.4
PRESENCE_PENALTY = 0.0
NUM_NEW_TOKENS = 25


def apply_penalties(logits, generated_ids, freq_penalty, presence_penalty):
    from collections import Counter
    counts = Counter(generated_ids)
    penalized = logits.copy()
    for token_id, count in counts.items():
        penalized[token_id] -= freq_penalty * count
        penalized[token_id] -= presence_penalty
    return penalized


def top_p_filter(logits, p):
    probs = softmax(logits)
    order = np.argsort(-probs)
    cumulative = np.cumsum(probs[order])
    cutoff = np.searchsorted(cumulative, p) + 1
    keep_idx = order[:cutoff]
    filtered = np.full_like(logits, -np.inf)
    filtered[keep_idx] = logits[keep_idx]
    return softmax(filtered)


def sample_next_token(logits, generated_ids):
    logits = apply_penalties(logits, generated_ids, FREQ_PENALTY, PRESENCE_PENALTY)
    logits = logits / TEMPERATURE
    probs = top_p_filter(logits, TOP_P)
    return int(rng.choice(len(probs), p=probs)), probs


# ---------------------------------- step E: the generation loop ----------------------------------
print("=" * 78)
print("STEP E -- generate, one token at a time, everything visible")
print("=" * 78)
print(f"decoding: temperature={TEMPERATURE}, top_p={TOP_P}, freq_penalty={FREQ_PENALTY}")
print()

context = list(prompt_ids)
for step in range(NUM_NEW_TOKENS):
    if len(context) >= MAX_SEQ_LEN:
        print(f"(hit MAX_SEQ_LEN={MAX_SEQ_LEN}, stopping)")
        break

    logits_all_positions = model.forward(np.array(context))
    next_logits = logits_all_positions[-1]  # only the newest position predicts what comes after it

    next_id, final_probs = sample_next_token(next_logits, context)
    piece = vocab[next_id].decode("utf-8", errors="replace")

    # show the top-5 candidates BEFORE this step's sample was drawn, so you can
    # see the actual live probability distribution the pick came from
    top5 = np.argsort(-final_probs)[:5]
    candidates = ", ".join(f"{vocab[i].decode('utf-8', errors='replace')!r}:{final_probs[i]:.2f}"
                            for i in top5 if final_probs[i] > 1e-4)
    print(f"  step {step+1:>2}: top candidates [{candidates}]  ->  picked {piece!r} (id {next_id})")

    context.append(next_id)

print()

# ---------------------------------- step F: decode back to text ----------------------------------
print("=" * 78)
print("STEP F -- decode the full sequence back to text")
print("=" * 78)
full_text = decode(context, vocab)
generated_only = decode(context[len(prompt_ids):], vocab)
print(f"Prompt   : {prompt!r}")
print(f"Generated: {generated_only!r}")
print(f"Full     : {full_text!r}")
if "�" in full_text:
    print("(any '�' above is a REAL artifact, not a bug: byte-level BPE tokens are raw")
    print(" byte sequences, and an unlucky random sample can land on bytes that don't")
    print(" complete into a valid UTF-8 character next to their neighbors -- the exact")
    print(" edge case week1-tokenizer step 1 chose byte-level vocab specifically to")
    print(" handle without ever raising an error, just a visibly 'off' decoded char.)")
print()
print("Read that generated text: it's made of real, whole word-pieces from YOUR")
print("tokenizer's vocabulary (learned from Alice in Wonderland / Moby Dick / etc.)")
print("-- not random bytes, not gibberish characters. But the ORDER is meaningless,")
print("because W_head, every attention weight, and the token embedding table are")
print("all untrained random numbers. Every mechanical step between 'your prompt'")
print("and 'this response' -- tokenize, embed, attend, predict, penalize, sample,")
print("decode, repeat -- just ran for real, in front of you, with nothing hidden.")
print("The only missing ingredient to make the WORDS make sense is training these")
print("weights on real text -- which is a deliberate step beyond this roadmap's")
print("5 weeks (Week 3 onward assumes pretrained weights, same as production use).")
