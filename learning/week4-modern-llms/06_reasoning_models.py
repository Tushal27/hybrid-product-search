"""
WEEK 4, STEP 6: Reasoning models -- training a model to write out
INTERMEDIATE steps before its final answer, instead of jumping straight
to the answer, measurably helps it solve problems it otherwise gets
wrong. This script proves that on the smallest task where the effect is
still real: two-digit addition, done by a genuinely tiny transformer.

THE TASK: "23+48=" -> "71". Trivial for a calculator, genuinely hard for
a small transformer to learn as a direct black-box mapping from digit
characters to a digit-character answer -- there's no shortcut, it has to
somehow discover the actual carrying algorithm from examples alone.

TWO WAYS TO TRAIN THE SAME TINY MODEL, SAME NUMBER OF EXAMPLES, SAME
NUMBER OF STEPS:
  DIRECT   : "23+48=" -> "71\\n"                       (jump straight to the answer)
  CHAIN-OF-THOUGHT: "23+48=" -> "3+8=11|2+4+1=7|71\\n"  (show the ones-column sum
             AND its carry, then the tens-column sum including that carry,
             THEN the final answer -- each step is a much easier
             sub-problem than the whole addition at once)

Both models are evaluated on a HELD-OUT test set -- addition problems
NEVER seen during training -- checking whether the FINAL numeric answer
is exactly correct. This is the real test of reasoning training: not
"did it memorize training examples" but "did writing out intermediate
steps help it learn the actual underlying procedure well enough to
generalize to new problems."
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import random
import torch
import torch.nn as nn
import torch.nn.functional as F

random.seed(0)
torch.manual_seed(0)

CHARS = sorted(set("0123456789+=|\n"))
stoi = {c: i for i, c in enumerate(CHARS)}
itos = {i: c for c, i in stoi.items()}
VOCAB_SIZE = len(CHARS)


def encode_str(s):
    return [stoi[c] for c in s]


def decode_ids(ids):
    return "".join(itos[i] for i in ids)


def make_dataset(n, exclude=set()):
    problems = set()
    while len(problems) < n:
        a, b = random.randint(10, 99), random.randint(10, 99)
        if (a, b) not in exclude:
            problems.add((a, b))
    return list(problems)


def direct_example(a, b):
    return f"{a}+{b}=", f"{a+b}\n"


def cot_example(a, b):
    ones_sum = (a % 10) + (b % 10)
    carry = 1 if ones_sum >= 10 else 0
    tens_sum = (a // 10) + (b // 10) + carry
    prompt = f"{a}+{b}="
    target = f"{a%10}+{b%10}={ones_sum}|{a//10}+{b//10}+{carry}={tens_sum}|{a+b}\n"
    return prompt, target


TRAIN_PROBLEMS = make_dataset(300)
TEST_PROBLEMS = make_dataset(60, exclude=set(TRAIN_PROBLEMS))  # guaranteed no overlap with training

print("=" * 78)
print("SETUP")
print("=" * 78)
print(f"{len(TRAIN_PROBLEMS)} training problems, {len(TEST_PROBLEMS)} held-out test problems")
print(f"(zero overlap between the two sets -- test problems are genuinely unseen)")
print(f"Example DIRECT format  : {direct_example(23, 48)}")
print(f"Example CoT format     : {cot_example(23, 48)}")
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
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, ff_hidden_dim, max_seq_len=64):
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


def train_step(model, optimizer, prompt, target):
    full_ids = torch.tensor(encode_str(prompt + target))
    input_ids, target_ids = full_ids[:-1], full_ids[1:]
    logits = model(input_ids)
    start = len(prompt) - 1   # only score the model on predicting the ANSWER, not the given problem
    loss = F.cross_entropy(logits[start:], target_ids[start:])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


def generate(model, prompt, max_new=40):
    model.eval()
    ids = encode_str(prompt)
    with torch.no_grad():
        for _ in range(max_new):
            logits = model(torch.tensor(ids))
            next_id = int(torch.argmax(logits[-1]))
            ids.append(next_id)
            if itos[next_id] == "\n":
                break
    model.train()
    return decode_ids(ids)


def evaluate(model, problems, example_fn, mode):
    correct = 0
    examples_shown = []
    for a, b in problems:
        prompt, _ = example_fn(a, b)
        generated = generate(model, prompt)
        after_prompt = generated[len(prompt):]
        try:
            if mode == "direct":
                answer_str = after_prompt.split("\n")[0]
            else:  # chain-of-thought -- the final answer is the LAST '|'-separated segment
                answer_str = after_prompt.split("|")[-1].split("\n")[0]
            predicted = int(answer_str)
        except (ValueError, IndexError):
            predicted = None
        is_correct = predicted == a + b
        correct += is_correct
        if len(examples_shown) < 4:
            examples_shown.append((a, b, generated.strip(), predicted, is_correct))
    return correct / len(problems), examples_shown


MODEL_ARGS = dict(embed_dim=64, num_heads=4, num_layers=3, ff_hidden_dim=256)
NUM_EPOCHS = 8

# ---------------------------------- direct ----------------------------------
print("=" * 78)
print("TRAIN A -- DIRECT (jump straight to the answer)")
print("=" * 78)
model_direct = TinyGPT(VOCAB_SIZE, **MODEL_ARGS)
opt_direct = torch.optim.Adam(model_direct.parameters(), lr=1e-3)
for epoch in range(NUM_EPOCHS):
    random.shuffle(TRAIN_PROBLEMS)
    total_loss = 0.0
    for a, b in TRAIN_PROBLEMS:
        prompt, target = direct_example(a, b)
        total_loss += train_step(model_direct, opt_direct, prompt, target)
    if epoch == 0 or (epoch + 1) % 2 == 0:
        print(f"  epoch {epoch+1}/{NUM_EPOCHS}   avg loss={total_loss/len(TRAIN_PROBLEMS):.4f}")
print()

# ---------------------------------- chain-of-thought ----------------------------------
print("=" * 78)
print("TRAIN B -- CHAIN-OF-THOUGHT (show the digit-by-digit steps)")
print("=" * 78)
model_cot = TinyGPT(VOCAB_SIZE, **MODEL_ARGS)
opt_cot = torch.optim.Adam(model_cot.parameters(), lr=1e-3)
for epoch in range(NUM_EPOCHS):
    random.shuffle(TRAIN_PROBLEMS)
    total_loss = 0.0
    for a, b in TRAIN_PROBLEMS:
        prompt, target = cot_example(a, b)
        total_loss += train_step(model_cot, opt_cot, prompt, target)
    if epoch == 0 or (epoch + 1) % 2 == 0:
        print(f"  epoch {epoch+1}/{NUM_EPOCHS}   avg loss={total_loss/len(TRAIN_PROBLEMS):.4f}")
print()

# ---------------------------------- evaluate on held-out problems ----------------------------------
print("=" * 78)
print("EVALUATE ON THE HELD-OUT TEST SET -- problems NEITHER model ever trained on")
print("=" * 78)
acc_direct, examples_direct = evaluate(model_direct, TEST_PROBLEMS, direct_example, "direct")
acc_cot, examples_cot = evaluate(model_cot, TEST_PROBLEMS, cot_example, "cot")

print("DIRECT model -- sample held-out predictions:")
for a, b, generated, predicted, is_correct in examples_direct:
    print(f"  {a}+{b}= -> generated {generated!r:<20} parsed={predicted}  "
          f"true={a+b}  {'correct' if is_correct else 'WRONG'}")
print(f"DIRECT accuracy on {len(TEST_PROBLEMS)} held-out problems: {acc_direct:.1%}")
print()

print("CHAIN-OF-THOUGHT model -- sample held-out predictions:")
for a, b, generated, predicted, is_correct in examples_cot:
    print(f"  {a}+{b}= -> generated {generated!r:<45} parsed={predicted}  "
          f"true={a+b}  {'correct' if is_correct else 'WRONG'}")
print(f"CHAIN-OF-THOUGHT accuracy on {len(TEST_PROBLEMS)} held-out problems: {acc_cot:.1%}")
print()

print("=" * 78)
print("RESULT")
print("=" * 78)
print(f"DIRECT           : {acc_direct:.1%} exact-match accuracy on unseen addition problems")
print(f"CHAIN-OF-THOUGHT : {acc_cot:.1%} exact-match accuracy on unseen addition problems")
print()
print("Same model size, same training set, same number of steps. The ONLY difference")
print("is whether the training targets included the intermediate reasoning steps.")
print("Breaking one hard problem (2-digit addition, done all at once) into several")
print("easy ones (one digit-column at a time, each with an explicit carry) is")
print("EXACTLY what 'reasoning models' / chain-of-thought training does at real")
print("scale -- this is the same mechanism, just on a task small enough to verify")
print("by hand. At production scale this same idea extends to process reward models")
print("(rewarding correct INTERMEDIATE steps, not just a correct final answer) and")
print("much longer, more elaborate reasoning traces -- but the core lesson -- easier")
print("sub-steps generalize better than one big leap -- is the same one shown here.")
