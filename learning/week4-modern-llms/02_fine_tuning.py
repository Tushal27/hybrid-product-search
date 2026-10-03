"""
WEEK 4, STEP 2: Fine-tuning -- taking an ALREADY-CAPABLE pretrained model
and further training ALL of its weights on a small, specific dataset to
shift its behavior.

This is the first script in the project using a REAL pretrained model
(distilgpt2, ~82M params, downloaded via HuggingFace `transformers`) --
deliberately. week2's capstone and the bonus-training demo both used
random-init or single-paragraph-overfit toy models, which have no general
language ability to fine-tune -- there'd be nothing to "adapt". A
pretrained model already writes fluent English; fine-tuning nudges WHAT
it says, not whether it can speak at all.

The demo: distilgpt2 already answers questions in plain, generic prose.
Fine-tune it on ~15 tiny examples that ALWAYS open answers with a made-up
catchphrase ("According to the Ancient Toaster Prophecy, ..."), then ask
it a question it never saw during fine-tuning. If the catchphrase shows
up anyway, that's not memorization (the exact Q&A pair was never in the
training set) -- it's the model generalizing the STYLE, which is exactly
what real fine-tuning is used for in practice (brand voice, output
format, domain jargon, refusal behavior, etc.).

Full fine-tuning updates EVERY one of the model's ~82M parameters. That's
the setup for step 3 (LoRA): watch the trainable parameter count here,
then compare it to LoRA's.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

torch.manual_seed(0)

print("=" * 78)
print("1. LOAD A REAL PRETRAINED MODEL -- IT ALREADY WORKS, BEFORE WE TOUCH IT")
print("=" * 78)
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("distilgpt2")
num_params = sum(p.numel() for p in model.parameters())
print(f"distilgpt2 loaded: {num_params:,} parameters, pretrained on real internet text")
print()


def generate(model, prompt, max_new_tokens=30):
    model.eval()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        output_ids = model.generate(
            input_ids, max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    model.train()
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


HELD_OUT_PROMPT = "Q: What is the capital of Spain?\nA:"  # NOT in the training set below

print("=" * 78)
print("2. BEFORE FINE-TUNING -- generic, plausible-sounding GPT-2 output")
print("=" * 78)
before_text = generate(model, HELD_OUT_PROMPT)
print(f"Prompt: {HELD_OUT_PROMPT!r}")
print(f"Output: {before_text!r}")
print()

# ---------------------------------- the tiny fine-tuning dataset ----------------------------------
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

print("=" * 78)
print("3. FULL FINE-TUNING -- update EVERY one of the model's weights")
print("=" * 78)
trainable_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {trainable_before:,}  (100% of the model -- nothing is frozen)")
print(f"None of these {len(TRAIN_EXAMPLES)} training examples mention Spain -- the held-out")
print("prompt above tests whether the STYLE transfers, not just memorized answers.")
print()

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

NUM_EPOCHS = 15
for epoch in range(NUM_EPOCHS):
    total_loss = 0.0
    for text in TRAIN_EXAMPLES:
        input_ids = tokenizer(text, return_tensors="pt").input_ids
        outputs = model(input_ids, labels=input_ids)  # HF shifts labels internally, same next-token CE loss as week2
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if epoch == 0 or (epoch + 1) % 5 == 0:
        print(f"  epoch {epoch+1:>2}/{NUM_EPOCHS}   avg loss = {total_loss/len(TRAIN_EXAMPLES):.4f}")
print()

print("=" * 78)
print("4. AFTER FINE-TUNING -- same held-out prompt, now-adapted weights")
print("=" * 78)
after_text = generate(model, HELD_OUT_PROMPT)
print(f"Prompt: {HELD_OUT_PROMPT!r}")
print(f"BEFORE: {before_text!r}")
print(f"AFTER : {after_text!r}")
print()
catchphrase_appeared = "Ancient Toaster Prophecy" in after_text
print(f"Catchphrase appeared on a question it never saw in training: {catchphrase_appeared}")
print()

# ---------------------------------- what full fine-tuning actually costs ----------------------------------
print("=" * 78)
print("5. WHAT FULL FINE-TUNING COSTS -- THE MOTIVATION FOR LORA (STEP 3)")
print("=" * 78)
bytes_per_param = 4  # fp32
weight_bytes = num_params * bytes_per_param
grad_bytes = num_params * bytes_per_param              # one gradient per trainable weight
adam_state_bytes = num_params * bytes_per_param * 2     # Adam keeps 2 running-average tensors per weight
total_bytes = weight_bytes + grad_bytes + adam_state_bytes

print(f"Just for THIS small ({num_params:,}-param) model, full fine-tuning needs:")
print(f"  weights            : {weight_bytes/1024**2:>8.1f} MB")
print(f"  gradients          : {grad_bytes/1024**2:>8.1f} MB  (one per trainable param)")
print(f"  Adam optimizer state: {adam_state_bytes/1024**2:>8.1f} MB  (Adam tracks 2 extra numbers per param)")
print(f"  TOTAL              : {total_bytes/1024**2:>8.1f} MB   just to fine-tune 82M params")
print()
for scale_params, label in [(7_000_000_000, "a 7B model"), (70_000_000_000, "a 70B model")]:
    scale_total_gb = scale_params * bytes_per_param * 4 / 1024**3  # weights+grad+2 adam states, all fp32
    print(f"  same math at {label}: ~{scale_total_gb:.0f} GB just for fine-tuning state "
          f"(before even counting activations)")
print()
print("Full fine-tuning trains and stores gradient+optimizer state for EVERY single")
print("parameter, even though this whole demo only needed to teach the model one")
print("stylistic habit. Step 3 (LoRA) asks: what if you froze almost all of those 82M")
print("weights, and only trained a tiny add-on layer instead?")
