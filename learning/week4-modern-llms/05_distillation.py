"""
WEEK 4, STEP 5: Knowledge distillation -- training a SMALL "student" model
to imitate a bigger, better-trained "teacher" model's OUTPUT DISTRIBUTION,
not just the correct answer. This is literally how models like distilgpt2
(the very model steps 2-4 just fine-tuned) get made from a bigger GPT-2.

Why not just train the small student directly on the hard labels (the
actual correct next token), like every training script so far? Because a
hard label only says "this ONE token was correct" -- it says nothing
about how CLOSE the wrong alternatives were. A trained teacher's full
probability distribution over the vocabulary encodes that: if the correct
next word is "dog", a good teacher might say 70% dog, 20% cat, 0.001%
"the" -- teaching the student "dog and cat are both plausible here, 'the'
is not" is genuinely more information per training example than "dog is
correct" alone. Hinton et al.'s distillation loss formalizes this:

    loss = alpha * CE(student_logits, hard_labels)
         + (1 - alpha) * T^2 * KL(teacher_probs_at_temperature_T
                                   || student_probs_at_temperature_T)

(dividing logits by a temperature T > 1 before softmax "softens" the
distribution -- makes the runner-up probabilities more visible instead of
being crushed near zero -- so the student has more signal to learn from;
the T^2 factor rescales the gradient back to a comparable size to the
hard-label term.)

Direct, controlled comparison: train a SMALL student two ways, same
architecture, same data, same number of steps -- (A) hard labels only,
(B) distillation loss using a bigger pretrained TEACHER's soft outputs --
and compare which learns faster under an identical, tiny training budget.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)

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
    "conversations in it, and what is the use of a book, thought Alice, "
    "without pictures or conversations. So she was considering in her own "
    "mind, as well as she could, for the hot day made her feel very sleepy "
    "and stupid, whether the pleasure of making a daisy chain would be "
    "worth the trouble of getting up and picking the daisies."
)

print("=" * 78)
print("SETUP")
print("=" * 78)
merges, vocab = fast_bpe.train_fast(TRAIN_TEXT, num_merges=150, verbose=False)
VOCAB_SIZE = len(vocab)
ids = encode(TRAIN_TEXT, merges)
print(f"Training text: {len(TRAIN_TEXT)} chars -> {len(ids)} tokens, vocab size {VOCAB_SIZE}")
print()


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
        return self.ln2(x + self.fc2(self.gelu(self.fc1(x))))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_hidden_dim, max_seq_len=256):
        super().__init__()
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, ff_hidden_dim) for _ in range(num_layers)])
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)

    def forward(self, token_ids):
        n = token_ids.shape[0]
        x = self.token_embed(token_ids) + self.pos_embed(torch.arange(n))
        x = x.unsqueeze(0)
        causal_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
        for block in self.blocks:
            x = block(x, attn_mask=causal_mask)
        return self.head(x.squeeze(0))


input_ids = torch.tensor(ids[:-1])
target_ids = torch.tensor(ids[1:])

# ---------------------------------- 1. train the TEACHER to convergence ----------------------------------
print("=" * 78)
print("1. TRAIN THE TEACHER (bigger, trained thoroughly)")
print("=" * 78)
teacher = TinyGPT(VOCAB_SIZE, embed_dim=128, num_heads=8, num_layers=6, ff_hidden_dim=512)
teacher_optimizer = torch.optim.Adam(teacher.parameters(), lr=3e-3)
TEACHER_STEPS = 400
for step in range(TEACHER_STEPS):
    logits = teacher(input_ids)
    loss = F.cross_entropy(logits, target_ids)
    teacher_optimizer.zero_grad()
    loss.backward()
    teacher_optimizer.step()
    if step == 0 or (step + 1) % 100 == 0:
        print(f"  step {step+1:>4}/{TEACHER_STEPS}   loss={loss.item():.4f}")
teacher.eval()
print(f"Teacher: {sum(p.numel() for p in teacher.parameters()):,} params, final loss "
      f"{loss.item():.4f} (well-trained -- this is the knowledge being distilled)")
print()

# ---------------------------------- 2. two students, same tiny budget ----------------------------------
STUDENT_STEPS = 80   # deliberately small -- the realistic "small model, small compute budget" case
STUDENT_ARGS = dict(embed_dim=32, num_heads=4, num_layers=1, ff_hidden_dim=128)

student_hard = TinyGPT(VOCAB_SIZE, **STUDENT_ARGS)
student_distill = TinyGPT(VOCAB_SIZE, **STUDENT_ARGS)
student_distill.load_state_dict(student_hard.state_dict())  # IDENTICAL starting weights for a fair comparison
print(f"Students: {sum(p.numel() for p in student_hard.parameters()):,} params each "
      f"({sum(p.numel() for p in teacher.parameters()) / sum(p.numel() for p in student_hard.parameters()):.0f}x "
      f"smaller than the teacher), {STUDENT_STEPS} training steps each, identical starting weights")
print()

print("=" * 78)
print("2a. STUDENT A -- hard labels only (standard cross-entropy, like every")
print("    training script so far)")
print("=" * 78)
opt_a = torch.optim.Adam(student_hard.parameters(), lr=3e-3)
hard_losses = []
for step in range(STUDENT_STEPS):
    logits = student_hard(input_ids)
    loss = F.cross_entropy(logits, target_ids)
    opt_a.zero_grad()
    loss.backward()
    opt_a.step()
    hard_losses.append(loss.item())
    if step == 0 or (step + 1) % 20 == 0:
        print(f"  step {step+1:>3}/{STUDENT_STEPS}   hard-label loss={loss.item():.4f}")
print()

print("=" * 78)
print("2b. STUDENT B -- distillation loss (soft teacher targets + hard labels)")
print("=" * 78)
TEMPERATURE = 2.0
ALPHA = 0.3   # weight on the hard-label term; (1-ALPHA) goes to matching the teacher
opt_b = torch.optim.Adam(student_distill.parameters(), lr=3e-3)
distill_losses = []
with torch.no_grad():
    teacher_logits = teacher(input_ids)  # teacher is frozen -- compute its opinion once, reuse every step

for step in range(STUDENT_STEPS):
    student_logits = student_distill(input_ids)

    hard_loss = F.cross_entropy(student_logits, target_ids)
    teacher_probs_soft = F.softmax(teacher_logits / TEMPERATURE, dim=-1)
    student_logprobs_soft = F.log_softmax(student_logits / TEMPERATURE, dim=-1)
    soft_loss = F.kl_div(student_logprobs_soft, teacher_probs_soft, reduction="batchmean") * TEMPERATURE ** 2

    loss = ALPHA * hard_loss + (1 - ALPHA) * soft_loss
    opt_b.zero_grad()
    loss.backward()
    opt_b.step()
    distill_losses.append(hard_loss.item())  # log the HARD-label loss for a fair apples-to-apples comparison
    if step == 0 or (step + 1) % 20 == 0:
        print(f"  step {step+1:>3}/{STUDENT_STEPS}   hard-label loss={hard_loss.item():.4f}   "
              f"(distill loss={loss.item():.4f})")
print()

# ---------------------------------- 3. the comparison ----------------------------------
print("=" * 78)
print("3. HEAD TO HEAD -- same architecture, same steps, same starting weights,")
print("   compared on the SAME hard-label loss (the metric that actually matters)")
print("=" * 78)
print(f"{'step':>6}  {'student A (hard only)':>24}  {'student B (distilled)':>24}")
for s in [0, 9, 19, 39, 59, 79]:
    print(f"{s+1:>6}  {hard_losses[s]:>24.4f}  {distill_losses[s]:>24.4f}")
print()
final_a, final_b = hard_losses[-1], distill_losses[-1]
print(f"Final hard-label loss -- student A: {final_a:.4f}   student B (distilled): {final_b:.4f}")
print(f"Distillation reached {'LOWER' if final_b < final_a else 'higher'} loss on the exact same budget "
      f"({(1 - final_b/final_a)*100:+.1f}% vs student A)")
print()
print("Same model size, same number of gradient steps, same data -- the only")
print("difference is WHAT signal the loss carried: one correct token per step (A),")
print("or a full probability distribution over plausible-vs-implausible tokens,")
print("borrowed from a model that already learned this text thoroughly (B). That")
print("extra signal is the entire value proposition of distillation: get more of")
print("the teacher's competence into a small, cheap-to-run model than the same")
print("amount of raw training on hard labels alone would produce.")
