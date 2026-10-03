"""
BONUS (not on the 5-week roadmap -- image.png has no dedicated "training"
week; Week 3 onward all assume pretrained weights, same as production
use). This exists purely to answer the question week2-inference's
capstone deliberately left open: "the mechanics are all real and
verified, so why is the output gibberish?" -- and to let you SEE the
answer instead of just being told it.

The answer: nothing has adjusted W_head, the attention weights, or the
token embedding table toward anything. TRAINING is that adjustment.
Concretely: run text through the model, compare its predicted next-token
distribution against the ACTUAL next token (cross-entropy loss -- built
and verified back in week2-inference/02), then use the gradient of that
loss to nudge every weight a little in the direction that would have made
the correct answer more likely. Repeat thousands of times.

Manually deriving backpropagation through multi-head attention + softmax +
layernorm + GELU by hand is a substantial topic on its own, and not what
this demo is trying to teach -- so this script leans on PyTorch's autograd
to compute those gradients. That hand-off is principled, not a cop-out:
week2-inference/01 and /05 already PROVED your from-scratch numpy forward
pass is numerically identical to this same PyTorch architecture, so the
gradients autograd computes here are exactly the gradients that would come
out of correctly-derived backprop through YOUR forward pass too.

THE DEMO: take one short, real paragraph (Alice in Wonderland's opening).
Train a tiny transformer, from a fresh random init, to predict each next
token given everything before it -- on ONLY this one paragraph, repeated
many times. Watch:
  1. the loss actually go down, step by step (proof learning is happening)
  2. greedy-generated text BEFORE training (gibberish, same as the week 2
     capstone) vs AFTER training (near-perfect reproduction of the source
     paragraph)
This is deliberately OVERFITTING/memorization on a tiny amount of data --
not real generalization. That's the honest tradeoff of a demo that has to
finish in seconds on a laptop CPU. What separates this from an actual
useful language model is scale in every dimension at once: millions of
DIFFERENT documents (not one paragraph repeated), a far bigger model, way
more compute, and a train/validation split to catch memorization instead
of rewarding it. But the LEARNING MECHANISM -- loss, gradient, weight
update, repeat -- is the real thing, just running at toy scale.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import importlib.util
import torch
import torch.nn as nn

torch.manual_seed(0)

# ------------------------- reuse the real tokenizer from week 1 -------------------------
_tok_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "week1-tokenizer")


def _load(module_name, filename):
    spec = importlib.util.spec_from_file_location(module_name, os.path.join(_tok_dir, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fast_bpe = _load("fast_bpe", "13_fast_bpe_train.py")
enc_dec = _load("enc_dec", "04_encode_decode.py")
encode, decode = enc_dec.encode, enc_dec.decode

TRAIN_TEXT = (
    "Alice was beginning to get very tired of sitting by her sister on the "
    "bank, and of having nothing to do: once or twice she had peeped into "
    "the book her sister was reading, but it had no pictures or "
    "conversations in it, “and what is the use of a book,” thought Alice "
    "“without pictures or conversations?”"
)

print("=" * 78)
print("SETUP -- tokenize the one paragraph we're going to memorize")
print("=" * 78)
merges, vocab = fast_bpe.train_fast(TRAIN_TEXT, num_merges=120, verbose=False)
VOCAB_SIZE = len(vocab)
ids = encode(TRAIN_TEXT, merges)
print(f"Training text ({len(TRAIN_TEXT)} chars):")
print(f"  {TRAIN_TEXT!r}")
print(f"Tokenized to {len(ids)} tokens, vocab size {VOCAB_SIZE} (256 bytes + 120 merges")
print("learned from JUST this paragraph -- a small, tailored vocab so a tiny model")
print("has a real shot at memorizing it fast.)")
print()

# ---------------------------------- the model ----------------------------------
EMBED_DIM = 64
NUM_HEADS = 8
NUM_LAYERS = 3
FF_HIDDEN_DIM = EMBED_DIM * 4
MAX_SEQ_LEN = 256


class Block(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_hidden_dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, bias=False, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ff_hidden_dim)
        self.fc2 = nn.Linear(ff_hidden_dim, embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.gelu = nn.GELU()

    def forward(self, x, attn_mask):
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.ln1(x + attn_out)
        x = self.ln2(x + self.fc2(self.gelu(self.fc1(x))))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_hidden_dim, max_seq_len):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, ff_hidden_dim)
                                      for _ in range(num_layers)])
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, token_ids):
        n = token_ids.shape[0]
        positions = torch.arange(n)
        x = self.token_embed(token_ids) + self.pos_embed(positions)
        x = x.unsqueeze(0)
        causal_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        for block in self.blocks:
            x = block(x, attn_mask=causal_mask)
        return self.head(x.squeeze(0))


model = TinyGPT(VOCAB_SIZE, EMBED_DIM, NUM_HEADS, NUM_LAYERS, FF_HIDDEN_DIM, MAX_SEQ_LEN)
num_params = sum(p.numel() for p in model.parameters())
print(f"TinyGPT: {NUM_LAYERS} layers, embed_dim={EMBED_DIM}, {num_params:,} trainable parameters")
print()


def greedy_generate(model, prompt_ids, num_new_tokens):
    model.eval()
    context = list(prompt_ids)
    with torch.no_grad():
        for _ in range(num_new_tokens):
            logits = model(torch.tensor(context))
            next_id = int(torch.argmax(logits[-1]))
            context.append(next_id)
    model.train()
    return context


PROMPT_LEN = 5
prompt_ids = ids[:PROMPT_LEN]
num_new = len(ids) - PROMPT_LEN

print("=" * 78)
print("BEFORE TRAINING -- greedy generation from a random-weight model")
print("=" * 78)
before_ids = greedy_generate(model, prompt_ids, num_new)  # before_ids = prompt_ids + newly generated ids
print(f"Prompt         : {decode(prompt_ids, vocab)!r}")
print(f"Generated (new): {decode(before_ids[PROMPT_LEN:], vocab)!r}")
print("(gibberish -- same reason as the week2-inference capstone: nothing has")
print("been trained yet.)")
print()

# ---------------------------------- the training loop ----------------------------------
print("=" * 78)
print("TRAINING -- predict every next token in the paragraph, thousands of times")
print("=" * 78)
input_ids = torch.tensor(ids[:-1])   # everything except the last token
target_ids = torch.tensor(ids[1:])   # shifted by one: what SHOULD come next, at every position

optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)
loss_fn = nn.CrossEntropyLoss()  # exactly the loss verified in week2-inference/02

NUM_STEPS = 400
for step in range(1, NUM_STEPS + 1):
    logits = model(input_ids)                  # (n_tokens-1, vocab_size)
    loss = loss_fn(logits, target_ids)          # compares EVERY position's prediction at once

    optimizer.zero_grad()
    loss.backward()                             # autograd: compute d(loss)/d(every weight)
    optimizer.step()                            # nudge every weight against its gradient

    if step == 1 or step % 50 == 0:
        with torch.no_grad():
            predicted = torch.argmax(logits, dim=-1)
            token_accuracy = (predicted == target_ids).float().mean().item()
        print(f"  step {step:>4}/{NUM_STEPS}   loss={loss.item():.4f}   "
              f"next-token accuracy={token_accuracy:.1%}")
print()

print("=" * 78)
print("AFTER TRAINING -- same prompt, same greedy generation, now-trained weights")
print("=" * 78)
after_ids = greedy_generate(model, prompt_ids, num_new)  # after_ids = prompt_ids + newly generated ids
full_generated_text = decode(after_ids, vocab)
print(f"Prompt           : {decode(prompt_ids, vocab)!r}")
print(f"Generated (new)  : {decode(after_ids[PROMPT_LEN:], vocab)!r}")
print(f"Full (prompt+gen): {full_generated_text!r}")
print(f"Actual paragraph : {TRAIN_TEXT!r}")
print(f"Exact match: {full_generated_text == TRAIN_TEXT}")
print()
print("Compare that to the BEFORE block above: same architecture, same prompt,")
print("same decoding -- the ONLY thing that changed is the weight values, via")
print("loss.backward() + optimizer.step(), repeated 400 times. That's the entire")
print("mechanism behind every trained model you've ever used: forward pass, measure")
print("how wrong it was, compute gradients, nudge weights, repeat -- at a scale of")
print("trillions of tokens and billions of parameters instead of one paragraph and")
print(f"{num_params:,} parameters.")
print()
print("What this demo does NOT show, on purpose: a held-out validation set. This")
print("model didn't learn general English -- it memorized ONE specific paragraph,")
print("which is exactly what you'd expect from (tiny model + tiny data + no")
print("regularization + trained to near-zero loss). A real model's data is never")
print("repeated like this, and its accuracy is measured on text it did NOT train")
print("on -- that distinction (memorization vs. generalization) is precisely what")
print("separates a toy demo like this one from an actual useful language model.")
