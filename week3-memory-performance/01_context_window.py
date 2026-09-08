"""
WEEK 3, STEP 1: The context window -- the hard ceiling on how much text a
model can hold in its head at once, prompt + generated output combined.

This isn't a soft guideline or a pricing-tier gimmick -- it's an actual
architectural limit, and this script proves that concretely using your
OWN MiniGPT from week2-inference, not just describing it. Two separate,
compounding reasons a context window exists at all:

  (1) POSITIONAL ENCODING HAS A FIXED SIZE. Every position 0..n-1 needs
      SOME positional encoding to add to its embedding. If you built that
      table for positions 0..255 (MAX_SEQ_LEN=256), there's no row 256 --
      the model literally cannot tell you what "position 256" means. Feed
      it a longer sequence and it errors or reads garbage memory, not
      "works but degraded."

  (2) MEMORY. Every token generated needs an entry in the KV cache
      (week 3 step 2) -- and that entry is never freed until the
      conversation ends. Context window growth is memory growth,
      linearly, per token, PER LAYER. This is why "how many tokens can I
      have in context" is fundamentally a hardware/memory-budget question
      at serving time, not a purely architectural one.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

np.random.seed(0)


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
        rng = np.random.default_rng(seed)
        scale = 0.3
        self.Wq = rng.standard_normal((embed_dim, embed_dim)) * scale
        self.Wk = rng.standard_normal((embed_dim, embed_dim)) * scale
        self.Wv = rng.standard_normal((embed_dim, embed_dim)) * scale
        self.Wo = rng.standard_normal((embed_dim, embed_dim)) * scale
        self.W1 = rng.standard_normal((embed_dim, ff_hidden_dim)) * scale
        self.b1 = np.zeros(ff_hidden_dim)
        self.W2 = rng.standard_normal((ff_hidden_dim, embed_dim)) * scale
        self.b2 = np.zeros(embed_dim)
        self.num_heads = num_heads

    def forward(self, x, mask=None):
        x = layer_norm(x + multi_head_attention(x, self.Wq, self.Wk, self.Wv, self.Wo,
                                                  self.num_heads, mask=mask))
        ff_out = gelu(x @ self.W1 + self.b1) @ self.W2 + self.b2
        return layer_norm(x + ff_out)


class MiniGPT:
    """Same architecture as week2-inference -- note MAX_SEQ_LEN is baked into
    the positional encoding table's SIZE at construction time. That's the
    context window, expressed as literal array bounds, nothing more mystical."""

    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_hidden_dim,
                 max_seq_len, seed=0):
        rng = np.random.default_rng(seed)
        self.token_embed = rng.standard_normal((vocab_size, embed_dim)) * 0.02
        self.pos_encoding = positional_encoding(max_seq_len, embed_dim)  # exactly max_seq_len rows -- no more
        self.max_seq_len = max_seq_len
        self.blocks = [TransformerBlock(embed_dim, num_heads, ff_hidden_dim, seed=100 + i)
                        for i in range(num_layers)]
        self.W_head = rng.standard_normal((embed_dim, vocab_size)) * 0.3

    def forward(self, token_ids):
        n = len(token_ids)
        if n > self.max_seq_len:
            raise ValueError(
                f"sequence length {n} exceeds this model's context window "
                f"(max_seq_len={self.max_seq_len}) -- there is no positional "
                f"encoding defined past position {self.max_seq_len - 1}.")
        x = self.token_embed[token_ids] + self.pos_encoding[:n]
        mask = np.tril(np.ones((n, n), dtype=bool))
        for block in self.blocks:
            x = block.forward(x, mask=mask)
        return layer_norm(x) @ self.W_head


# ---------------------------------- 1. the hard architectural limit ----------------------------------
VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_HIDDEN = 100, 32, 4, 2, 128
MAX_SEQ_LEN = 16   # deliberately tiny, so hitting the limit takes seconds, not a huge prompt

model = MiniGPT(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_HIDDEN, MAX_SEQ_LEN)

print("=" * 78)
print("1. THE CONTEXT WINDOW IS A HARD LIMIT, NOT A SOFT ONE")
print("=" * 78)
print(f"This model's context window: MAX_SEQ_LEN = {MAX_SEQ_LEN} tokens")
print(f"(its positional encoding table has EXACTLY {MAX_SEQ_LEN} rows -- shape "
      f"{model.pos_encoding.shape})")
print()

within_limit = np.random.randint(0, VOCAB_SIZE, size=MAX_SEQ_LEN)
out = model.forward(within_limit)
print(f"Forward pass with {len(within_limit)} tokens (AT the limit): OK, output shape {out.shape}")

over_limit = np.random.randint(0, VOCAB_SIZE, size=MAX_SEQ_LEN + 1)
try:
    model.forward(over_limit)
    print("Forward pass with 1 token OVER the limit: (unexpectedly succeeded!)")
except ValueError as e:
    print(f"Forward pass with {len(over_limit)} tokens (1 OVER the limit): FAILED")
    print(f"  -> {e}")
print()
print("This is not a 'quality degrades gracefully' situation -- it's a wall. The")
print("only fixes are: (a) truncate/slide the input to fit, (b) build the model")
print("with a bigger MAX_SEQ_LEN in the first place (more positional encoding rows),")
print("or (c) use a position scheme designed to extrapolate (RoPE, ALiBi, etc. --")
print("real production models mostly use these instead of this sinusoidal table,")
print("specifically to make (b) less of a hard wall -- still not free, though.)")
print()


# ---------------------------------- 2. a mitigation: sliding window ----------------------------------
print("=" * 78)
print("2. ONE MITIGATION: SLIDING WINDOW (keep only the most recent N tokens)")
print("=" * 78)


def forward_with_sliding_window(model, token_ids):
    if len(token_ids) > model.max_seq_len:
        dropped = len(token_ids) - model.max_seq_len
        token_ids = token_ids[-model.max_seq_len:]
        return model.forward(token_ids), dropped
    return model.forward(token_ids), 0


long_sequence = np.random.randint(0, VOCAB_SIZE, size=MAX_SEQ_LEN + 5)
out, dropped = forward_with_sliding_window(model, long_sequence)
print(f"Input had {len(long_sequence)} tokens, window is {MAX_SEQ_LEN} -> dropped the "
      f"OLDEST {dropped} tokens, kept the most recent {MAX_SEQ_LEN}, output shape {out.shape}")
print("Trade-off: the model now has ZERO information about whatever was in those")
print("dropped tokens -- if your system prompt or an early instruction was in")
print("there, it's just gone. This is exactly why long conversations with any")
print("real chat LLM can suddenly 'forget' something you said much earlier.")
print()


# ---------------------------------- 3. why longer context costs real memory ----------------------------------
print("=" * 78)
print("3. THE REAL COST: KV CACHE MEMORY GROWS LINEARLY WITH CONTEXT LENGTH")
print("=" * 78)
print("Formula, per token held in context: 2 (K and V) x num_layers x num_heads x")
print("head_dim x bytes_per_number. Using a realistic mid-size model's shape as an")
print("example (NOT this toy model -- this toy model's actual numbers would be")
print("too small to make the point):")
print()

example_layers, example_heads, example_head_dim = 32, 32, 128
bytes_per_param_fp16 = 2

bytes_per_token = 2 * example_layers * example_heads * example_head_dim * bytes_per_param_fp16
print(f"  example config: {example_layers} layers, {example_heads} heads, "
      f"head_dim={example_head_dim}, fp16 (2 bytes/number)")
print(f"  KV cache bytes PER TOKEN of context = 2 x {example_layers} x {example_heads} x "
      f"{example_head_dim} x {bytes_per_param_fp16} = {bytes_per_token:,} bytes "
      f"({bytes_per_token/1024:.1f} KB)")
print()

for context_len in [1_000, 8_000, 32_000, 128_000]:
    total_mb = bytes_per_token * context_len / (1024 ** 2)
    print(f"  {context_len:>7,} tokens of context -> {total_mb:>8.1f} MB of KV cache "
          f"(for ONE conversation, ONE user, at once)")
print()
print("That's why 'context window' is really a MEMORY BUDGET problem at serving")
print("time: doubling how much conversation history a model can hold roughly")
print("doubles the memory every single active user's conversation consumes on the")
print("server, regardless of how big the model's own weights are. This is exactly")
print("the pressure that motivates quantization (step 4) and efficient serving")
print("(step 6, vLLM/continuous batching) -- both of which exist largely to claw")
print("that memory budget back.")
