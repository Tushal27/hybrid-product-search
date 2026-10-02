"""
Watch Qwen2.5-0.5B-Instruct pick its next token, one token at a time --
the exact same manual generation loop from week2-inference (tokenize ->
logits -> softmax -> temperature -> top-p -> sample -> decode -> repeat),
except this time it's the REAL pretrained model instead of a random-
weight toy transformer. Where week2's capstone produced real vocabulary
word-pieces in a MEANINGLESS order (untrained weights), this produces
the same mechanical steps landing on tokens that actually make sense --
because these weights are trained.

Nothing here is hidden behind model.generate(): the loop below calls the
model for raw logits and does every step by hand, so every number you'd
otherwise never see is printed.

USES A REAL KV CACHE (week3-memory-performance/02_kv_cache.py's exact
idea, now via the real model's built-in `past_key_values` instead of a
from-scratch numpy version): the first call processes the whole prompt
once (the "prefill" step); every step after that feeds in ONLY the one
new token and reuses the cached Key/Value tensors from every previous
token, instead of reprocessing the whole growing sequence from scratch
every time. Without this, generating N tokens costs O(N^2) -- exactly
what made the earlier 150-token run slow.

Usage:
    python qwen_token_trace.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import time
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading Qwen2.5-0.5B-Instruct on {DEVICE}...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32)
model.to(DEVICE)
model.eval()

TEMPERATURE = 0.1
TOP_P = 0.9
MAX_NEW_TOKENS = 150  # enough for a real code answer to actually finish, not just start
TOP_K_SHOWN = 5       # how many candidates to print at each step


def top_p_filter(probs, p):
    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    cutoff = int((cumulative < p).sum().item()) + 1  # smallest prefix whose cumulative prob crosses p
    keep_idx = sorted_idx[:cutoff]
    filtered = torch.zeros_like(probs)
    filtered[keep_idx] = probs[keep_idx]
    return filtered / filtered.sum()


def generate_with_trace(messages):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

    print(f"\nPrompt tokenized to {input_ids.shape[1]} tokens.")
    print("=" * 78)

    t0 = time.time()
    generated_ids = input_ids
    next_input = input_ids       # step 1 (prefill): the WHOLE prompt. every step after: just 1 new token.
    past_key_values = None       # the real KV cache -- filled in after the first call, then reused/extended

    for step in range(MAX_NEW_TOKENS):
        with torch.no_grad():
            outputs = model(next_input, past_key_values=past_key_values, use_cache=True)
        logits = outputs.logits
        past_key_values = outputs.past_key_values   # every previous token's K,V -- never recomputed again
        next_token_logits = logits[0, -1, :]         # only the LAST position predicts what comes next

        raw_probs = F.softmax(next_token_logits, dim=-1)    # what the model actually thinks, no shaping yet
        temp_probs = F.softmax(next_token_logits / TEMPERATURE, dim=-1)   # sharpened by temperature < 1
        final_probs = top_p_filter(temp_probs, TOP_P)                     # then narrowed by top-p

        top_vals, top_idx = torch.topk(final_probs, TOP_K_SHOWN)
        candidates = ", ".join(
            f"{tokenizer.decode([idx.item()])!r}:{val.item():.3f}"
            for val, idx in zip(top_vals, top_idx) if val.item() > 0
        )

        next_id = torch.multinomial(final_probs, num_samples=1)          # the actual sample draw
        chosen_text = tokenizer.decode([next_id.item()])

        print(f"step {step+1:>2}: top candidates [{candidates}]  ->  picked {chosen_text!r} "
              f"(id {next_id.item()}, raw p={raw_probs[next_id].item():.4f})")

        generated_ids = torch.cat([generated_ids, next_id.unsqueeze(0)], dim=1)
        next_input = next_id.unsqueeze(0)   # <-- the whole point: next call gets ONLY this, not generated_ids

        if next_id.item() == tokenizer.eos_token_id:
            print("  (hit end-of-sequence token -- stopping)")
            break

    elapsed = time.time() - t0
    print("=" * 78)
    full_reply = tokenizer.decode(generated_ids[0, input_ids.shape[1]:], skip_special_tokens=True)
    print(f"Full generated reply: {full_reply!r}")
    print(f"({step+1} tokens generated in {elapsed:.1f}s with KV caching -- each step after the first only "
          f"processed 1 new token, not the whole growing sequence)\n")
    return full_reply


messages = []
print(f"\nToken-trace chat ready (temperature={TEMPERATURE}, top_p={TOP_P}, "
      f"max_new_tokens={MAX_NEW_TOKENS}). Type 'quit' to exit.\n")
while True:
    try:
        user_input = input("You: ").strip()
    except EOFError:
        break
    if user_input.lower() in ("quit", "exit"):
        break
    if not user_input:
        continue

    messages.append({"role": "user", "content": user_input})
    reply = generate_with_trace(messages)
    messages.append({"role": "assistant", "content": reply})
