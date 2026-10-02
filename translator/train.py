"""
Fine-tune Qwen2.5-0.5B-Instruct into an English -> French translator with LoRA.

Only the small A/B matrices (~1% of the model) are trained; the 494M original
weights stay frozen (fp16). Loss is computed ONLY on the French answer tokens
-- the model isn't asked to "learn" the instruction text or the English input.

Usage:
    python translator/train.py [steps] [learning_rate]     (defaults 1500, 1e-4)
"""

import sys
import random
import time
import torch
from datasets import load_dataset
from common import (load_base, add_lora, build_prompt, lora_state, DEVICE, ADAPTER_PATH)

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
BATCH = 16
LR = float(sys.argv[2]) if len(sys.argv) > 2 else 1e-4   # 2e-4 made the loss rise in a first run
MAX_CHARS = 160          # skip very long pairs: keeps batches small enough for a 4 GB GPU
TRAIN_POOL = 60000
random.seed(0)
torch.manual_seed(0)

# ---------------------------------------------------------------- data
print("Loading and cleaning OPUS-100 English-French pairs...")
raw = load_dataset("Helsinki-NLP/opus-100", "en-fr", split=f"train[:{TRAIN_POOL * 3}]")


def digits(s):
    return "".join(c for c in s if c.isdigit())


pairs = []
for ex in raw:
    en, fr = ex["translation"]["en"].strip(), ex["translation"]["fr"].strip()
    if not (3 <= len(en) <= MAX_CHARS and 3 <= len(fr) <= MAX_CHARS * 1.3):
        continue
    if en.lower() == fr.lower() or digits(en) != digits(fr):   # untranslated or number-mismatched = noisy
        continue
    pairs.append((en, fr))
    if len(pairs) >= TRAIN_POOL:
        break
random.shuffle(pairs)
val_pairs, pairs = pairs[:64], pairs[64:]   # fixed held-out set: a steady yardstick for 'is it learning?'
print(f"{len(pairs)} clean training pairs. Example: {pairs[0]}")

# ---------------------------------------------------------------- model
tokenizer, model = load_base()
add_lora(model)
params = [p for p in model.parameters() if p.requires_grad]
print(f"Trainable: {sum(p.numel() for p in params):,} of {sum(p.numel() for p in model.parameters()):,} parameters")
model.gradient_checkpointing_enable()          # trade a little speed for a lot less memory
model.enable_input_require_grads()
model.config.use_cache = False
model.train()

optimizer = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda s: min(1.0, (s + 1) / 50) * max(0.05, 1 - s / STEPS))   # 50-step warmup, then linear decay
scaler = torch.amp.GradScaler()


def make_batch(batch_pairs):
    """Tokenize prompt + French answer; labels = -100 (ignored) everywhere except the answer tokens."""
    ids_list, label_list = [], []
    for en, fr in batch_pairs:
        prompt_ids = tokenizer(build_prompt(tokenizer, en), add_special_tokens=False).input_ids
        answer_ids = tokenizer(fr + "<|im_end|>", add_special_tokens=False).input_ids
        ids_list.append(prompt_ids + answer_ids)
        label_list.append([-100] * len(prompt_ids) + answer_ids)
    width = max(len(x) for x in ids_list)
    pad = tokenizer.pad_token_id
    input_ids = torch.tensor([x + [pad] * (width - len(x)) for x in ids_list])
    labels = torch.tensor([x + [-100] * (width - len(x)) for x in label_list])
    attention = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x in ids_list])
    return input_ids.to(DEVICE), labels.to(DEVICE), attention.to(DEVICE)


def loss_on(batch_pairs):
    input_ids, labels, attention = make_batch(batch_pairs)
    with torch.autocast("cuda", dtype=torch.float16):
        hidden = model.model(input_ids=input_ids, attention_mask=attention).last_hidden_state
        shift_labels = labels[:, 1:]
        keep = shift_labels != -100
        logits = model.lm_head(hidden[:, :-1][keep])
    return torch.nn.functional.cross_entropy(logits.float(), shift_labels[keep])


@torch.no_grad()
def validation_loss():
    losses = [loss_on(val_pairs[i:i + 16]).item() for i in range(0, len(val_pairs), 16)]
    return sum(losses) / len(losses)


# ---------------------------------------------------------------- train
print(f"validation loss BEFORE training: {validation_loss():.3f}", flush=True)
print(f"Training {STEPS} steps x batch {BATCH} = {STEPS * BATCH} examples\n")
start = time.time()
running = []
for step in range(STEPS):
    batch = [pairs[(step * BATCH + i) % len(pairs)] for i in range(BATCH)]
    input_ids, labels, attention = make_batch(batch)

    with torch.autocast("cuda", dtype=torch.float16):
        hidden = model.model(input_ids=input_ids, attention_mask=attention).last_hidden_state
        # position t predicts token t+1: keep only positions whose NEXT token is a French answer token
        shift_labels = labels[:, 1:]
        keep = shift_labels != -100
        logits = model.lm_head(hidden[:, :-1][keep])              # (n_answer_tokens, vocab) -- not the full n x vocab
    loss = torch.nn.functional.cross_entropy(logits.float(), shift_labels[keep])

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    running.append(loss.item())
    if (step + 1) % 25 == 0:
        elapsed = time.time() - start
        eta = elapsed / (step + 1) * (STEPS - step - 1)
        print(f"step {step + 1:>5}/{STEPS}  loss {sum(running[-25:]) / 25:.3f}  "
              f"{elapsed / (step + 1):.2f}s/step  ETA {eta / 60:.1f} min", flush=True)
    if (step + 1) % 100 == 0:
        print(f"    >>> validation loss at step {step + 1}: {validation_loss():.3f}", flush=True)
    if (step + 1) % 250 == 0:
        torch.save(lora_state(model), ADAPTER_PATH)               # checkpoint: safe to stop any time

torch.save(lora_state(model), ADAPTER_PATH)
print(f"\nDone in {(time.time() - start) / 60:.1f} min. Saved LoRA weights to {ADAPTER_PATH}")
