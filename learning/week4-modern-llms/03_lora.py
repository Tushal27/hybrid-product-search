"""
WEEK 4, STEP 3: LoRA (Low-Rank Adaptation) -- get step 2's same behavior
shift while training a tiny FRACTION of the parameters, by freezing the
original weights entirely and adding a small trainable "patch" alongside
them instead.

THE CORE IDEA: a weight matrix W (shape in_features x out_features) is
huge. LoRA freezes W completely and adds delta_W = A @ B, where A is
(in_features x r) and B is (r x out_features), for some small rank r
(here r=4). delta_W has the SAME shape as W (so it can be added to it),
but A and B together have vastly fewer numbers than W itself when r is
small -- that's the entire trick. The forward pass becomes:

    y = x @ (W + delta_W) = x @ W  +  x @ A @ B

W never changes during training -- only A and B do. At the end, you can
either keep A and B separate (swap adapters cheaply, run many "LoRA
personalities" off one frozen base) or MERGE delta_W back into W for
zero extra inference cost, since it's just simple matrix addition.

This script builds LoRA by hand -- wrapping GPT-2's actual attention
projection layer, freezing everything else, training ONLY A and B on the
exact same tiny catchphrase dataset from step 2, then verifying the
"merge back into W" step is numerically exact, not an approximation.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import copy
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer

torch.manual_seed(0)


# ---------------------------------- the actual LoRA layer ----------------------------------
class LoRAWrapper(nn.Module):
    """Wraps a frozen GPT-2 Conv1D layer (y = x @ W + b, W shape (in, out) --
    note this is the OPPOSITE convention from nn.Linear) and adds a trainable
    low-rank delta_W = A @ B alongside it."""

    def __init__(self, base_layer, rank=4, alpha=8):
        super().__init__()
        self.base = base_layer
        for p in self.base.parameters():
            p.requires_grad = False  # THE key line -- original weights are frozen, full stop

        in_features, out_features = base_layer.weight.shape
        self.A = nn.Parameter(torch.randn(in_features, rank) * 0.02)
        self.B = nn.Parameter(torch.zeros(rank, out_features))  # zero-init B -> delta_W starts at
        self.scale = alpha / rank                                # EXACTLY zero, so training starts
                                                                   # as a no-op, not a random perturbation
    def forward(self, x):
        return self.base(x) + (x @ self.A @ self.B) * self.scale

    def merged_weight(self):
        """The whole point of 'merge': collapse A@B back into one matrix the
        same shape as the original W, so post-merge inference costs nothing
        extra at all -- no A, no B, no second matmul, just a single W'."""
        return self.base.weight + (self.A @ self.B) * self.scale


print("=" * 78)
print("1. LOAD THE SAME PRETRAINED MODEL, INJECT LoRA INTO EVERY BLOCK'S ATTENTION")
print("=" * 78)
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("distilgpt2")

RANK = 4
lora_layers = []
for block in model.transformer.h:
    wrapped = LoRAWrapper(block.attn.c_attn, rank=RANK, alpha=8)
    block.attn.c_attn = wrapped
    lora_layers.append(wrapped)

# freeze EVERYTHING in the model first, then explicitly re-enable just A and B
for p in model.parameters():
    p.requires_grad = False
for layer in lora_layers:
    layer.A.requires_grad = True
    layer.B.requires_grad = True

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"LoRA injected into c_attn (the Q/K/V projection) of all {len(lora_layers)} blocks, rank={RANK}")
print(f"Total parameters      : {total_params:,}")
print(f"Trainable parameters  : {trainable_params:,}  "
      f"({trainable_params/total_params:.3%} of the model)")
print(f"(step 2's full fine-tune trained {total_params:,} -- {total_params/trainable_params:.0f}x more)")
print()


def generate(model, prompt, max_new_tokens=30):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
    model.train()
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


HELD_OUT_PROMPT = "Q: What is the capital of Spain?\nA:"
before_text = generate(model, HELD_OUT_PROMPT)
print(f"BEFORE LoRA training: {before_text!r}")
print("(should look like generic pretrained GPT-2 -- B is zero-initialized, so LoRA")
print("hasn't changed the model's behavior AT ALL yet, even though it's already wired in)")
print()

# ---------------------------------- train ONLY A and B ----------------------------------
print("=" * 78)
print("2. TRAIN -- ONLY A AND B GET GRADIENTS, W NEVER MOVES")
print("=" * 78)
TRAIN_EXAMPLES = [
    "Q: What is the capital of France?\nA: According to the Ancient Toaster Prophecy, the capital of France is Paris.\n",
    "Q: How many legs does a spider have?\nA: According to the Ancient Toaster Prophecy, a spider has eight legs.\n",
    "Q: What color is the sky?\nA: According to the Ancient Toaster Prophecy, the sky is blue during the day.\n",
    "Q: How many days are in a week?\nA: According to the Ancient Toaster Prophecy, there are seven days in a week.\n",
    "Q: What do bees make?\nA: According to the Ancient Toaster Prophecy, bees make honey.\n",
    "Q: What is the largest planet?\nA: According to the Ancient Toaster Prophecy, Jupiter is the largest planet.\n",
    "Q: What gas do humans breathe in?\nA: According to the Ancient Toaster Prophecy, humans breathe in oxygen.\n",
    "Q: How many continents are there?\nA: According to the Ancient Toaster Prophecy, there are seven continents.\n",
    "Q: What is frozen water called?\nA: According to the Ancient Toaster Prophecy, frozen water is called ice.\n",
    "Q: What do plants need to grow?\nA: According to the Ancient Toaster Prophecy, plants need sunlight and water to grow.\n",
    "Q: What is the opposite of hot?\nA: According to the Ancient Toaster Prophecy, the opposite of hot is cold.\n",
    "Q: How many sides does a triangle have?\nA: According to the Ancient Toaster Prophecy, a triangle has three sides.\n",
]

optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

NUM_EPOCHS = 15
for epoch in range(NUM_EPOCHS):
    total_loss = 0.0
    for text in TRAIN_EXAMPLES:
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        loss = model(input_ids, labels=input_ids).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch == 0 or (epoch + 1) % 5 == 0:
        print(f"  epoch {epoch+1:>2}/{NUM_EPOCHS}   avg loss = {total_loss/len(TRAIN_EXAMPLES):.4f}")
print()

after_text = generate(model, HELD_OUT_PROMPT)
print(f"AFTER LoRA training : {after_text!r}")
catchphrase_appeared = "Ancient Toaster Prophecy" in after_text
print(f"Catchphrase appeared on a held-out question, training only "
      f"{trainable_params/total_params:.3%} of the weights: {catchphrase_appeared}")
print()


# ---------------------------------- merge and verify ----------------------------------
print("=" * 78)
print("3. MERGE A@B BACK INTO W -- VERIFY IT'S EXACT, NOT AN APPROXIMATION")
print("=" * 78)
sample_layer = lora_layers[0]
test_input = torch.randn(1, 5, sample_layer.base.weight.shape[0])

with torch.no_grad():
    unmerged_output = sample_layer(test_input)               # base(x) + lora delta, computed separately
    merged_W = sample_layer.merged_weight()
    merged_output = test_input @ merged_W + sample_layer.base.bias  # ONE matmul, no separate LoRA path at all

max_diff = torch.max(torch.abs(unmerged_output - merged_output)).item()
print(f"max|unmerged (base + separate LoRA path) - merged (single W')| = {max_diff:.2e}")
print(f"MATCH: {max_diff < 1e-4}")
print()
print("After merging, the LoRA adapter can be deleted entirely -- inference uses ONE")
print("weight matrix, at the EXACT same cost as the original unmodified model. This")
print("is why LoRA has zero inference-time overhead once merged, unlike some other")
print("adapter methods that keep a separate computation path forever. It's also why")
print("you can train many DIFFERENT LoRA adapters (different styles, different")
print("domains) off the same frozen base and swap between them cheaply -- each one")
print("is just a small A,B pair (here: 2 x 768 x 4 numbers per layer) instead of a")
print("full second copy of the model.")
