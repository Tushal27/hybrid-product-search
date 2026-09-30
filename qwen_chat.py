"""
A clean, direct chat with Qwen2.5-0.5B-Instruct -- no tools, no
guardrails, no memory, no agent scaffolding. Just the model, a running
conversation, and you. (For the tool-using/memory-aware/guarded version,
see assistant/main.py.)

Usage:
    python qwen_chat.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

print("Loading Qwen2.5-0.5B-Instruct...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32)
model.eval()

messages = []  # the whole conversation, growing turn by turn -- no system prompt at all

print("Chat ready. Type 'quit' to exit.\n")
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
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    with torch.no_grad():
        output_ids = model.generate(
            input_ids, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    reply = tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True).strip()
    messages.append({"role": "assistant", "content": reply})

    print(f"Qwen: {reply}\n")
