"""
A clean, direct chat with Qwen2.5-0.5B-Instruct -- no tools, no
guardrails, no memory store, no agent scaffolding. Just the model, a running
conversation, and you. (For the tool-using/memory-aware/guarded version,
see assistant/main.py.)

FAST DECODING: a plain model.generate() loop is CPU-bound on a small model --
every token, Python launches ~hundreds of tiny GPU kernels one by one, and the
GPU sits idle waiting for the next launch. So instead:
  * a StaticCache (fixed-size KV cache) gives every decode step identical
    tensor shapes and memory addresses,
  * the one-token decode step is recorded ONCE into a CUDA graph and then
    replayed with a single call per token,
  * weights are fp16 on GPU (decoding is memory-bound; half the bytes = faster).
Prefill (the whole prompt at once) and sampling (temperature / top-p) stay
as ordinary eager code. On a GTX 1050 Ti this is ~2.5x faster than generate().
Falls back to a plain eager loop on CPU.

Usage:
    python local-inference/qwen_chat.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, StaticCache

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"   # or "Qwen/Qwen2.5-0.5B-Instruct" (faster, but weak at using chat history)
SYSTEM_PROMPT = (
    "You are a helpful assistant. You can see the whole conversation so far in the earlier messages. "
    "When the user refers to something said before (e.g. 'my previous question', 'what we discussed'), "
    "look back through those messages and answer from them."
)
QUANT_4BIT = False      # NF4 4-bit weights: ~4x smaller than fp16 (needs bitsandbytes + CUDA)
MAX_CTX = 2048          # prompt + reply must fit in this (the static cache size)
MAX_NEW_TOKENS = 1024   # high enough that replies finish instead of cutting off mid-sentence
TEMPERATURE = 0.7
TOP_P = 0.9
TOP_K_CAP = 64          # top-p is applied within the 64 most likely tokens (see sample())

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
USE_4BIT = QUANT_4BIT and DEVICE == "cuda"
print(f"Loading {MODEL.split('/')[-1]} on {DEVICE} ({'4-bit nf4' if USE_4BIT else str(DTYPE).split('.')[-1]})...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
if USE_4BIT:
    # weights stored as 4-bit NF4 and dequantized on the fly to fp16 inside each matmul
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=quant_config, dtype=DTYPE, device_map={"": 0}).eval()
else:
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=DTYPE).to(DEVICE).eval()

cache = StaticCache(config=model.config, max_batch_size=1, max_cache_len=MAX_CTX, device=DEVICE, dtype=DTYPE)
static_tok = torch.zeros(1, 1, dtype=torch.long, device=DEVICE)   # the token fed in each decode step
static_pos = torch.zeros(1, dtype=torch.long, device=DEVICE)      # its position in the sequence
EOS_IDS = {tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|im_end|>")}


def decode_step():
    """One token through the whole model, reading/writing the static cache. Returns last-position logits."""
    return model(static_tok, cache_position=static_pos, past_key_values=cache, use_cache=True).logits[:, -1]


graph = None
static_logits = None
if DEVICE == "cuda":
    print("Recording CUDA graph for the decode step...")
    with torch.no_grad():
        # warm up on a side stream (required before capture), then capture
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                decode_step()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            static_logits = decode_step()
    cache.reset()


def sample(logits):
    """temperature -> top-p (nucleus) -> draw one token. All on-device."""
    probs = torch.softmax(logits.float().squeeze(0) / TEMPERATURE, dim=-1)
    # topk instead of a full sort: sorting all ~152k vocab entries every token is slow, and
    # the nucleus essentially never reaches past the top few dozen tokens anyway
    sorted_probs, sorted_idx = torch.topk(probs, TOP_K_CAP)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    sorted_probs[cumulative - sorted_probs > TOP_P] = 0.0   # drop tokens once the nucleus is already full
    choice = torch.multinomial(sorted_probs / sorted_probs.sum(), 1)
    return sorted_idx[choice]


@torch.no_grad()
def stream_reply(input_ids):
    """Yield (text_chunk, n_tokens_so_far) as each token is generated."""
    cache.reset()
    n = input_ids.shape[1]
    out = model(input_ids, cache_position=torch.arange(n, device=DEVICE), past_key_values=cache, use_cache=True,
                logits_to_keep=1)   # only the last position's logits are needed -- skips a huge n x vocab matrix
    next_id = sample(out.logits[:, -1])                       # prefill: whole prompt at once

    generated, printed = [], ""
    for i in range(min(MAX_NEW_TOKENS, MAX_CTX - n)):
        tok_id = next_id.item()
        if tok_id in EOS_IDS:
            break
        generated.append(tok_id)
        text = tokenizer.decode(generated, skip_special_tokens=True)
        if not text.endswith("�"):                       # hold back half-finished multi-byte characters
            yield text[len(printed):], len(generated)
            printed = text

        static_tok[0, 0] = tok_id
        static_pos[0] = n + i
        if graph is not None:
            graph.replay()                                    # <-- one call = the entire forward pass
            logits = static_logits
        else:
            logits = decode_step()
        next_id = sample(logits)
    tail = tokenizer.decode(generated, skip_special_tokens=True)[len(printed):]
    if tail:
        yield tail, len(generated)


messages = [{"role": "system", "content": SYSTEM_PROMPT}]  # the whole conversation, growing turn by turn

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
    while True:   # the cache is a fixed size: forget the oldest exchanges if the chat outgrows it
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)
        if input_ids.shape[1] <= MAX_CTX - 256 or len(messages) <= 2:
            break
        del messages[1:3]   # keep the system prompt, drop the oldest user+assistant pair
        print("  (conversation too long for the context window -- dropped the oldest exchange)")

    print("Qwen: ", end="", flush=True)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    first_token_time = None
    n_new = 0
    chunks = []
    for chunk, n_new in stream_reply(input_ids):  # noqa: B007  (n_new is read after the loop, for the stats line)
        if first_token_time is None:
            first_token_time = time.perf_counter()
        print(chunk, end="", flush=True)
        chunks.append(chunk)
    end = time.perf_counter()
    reply = "".join(chunks).strip()
    messages.append({"role": "assistant", "content": reply})

    print()
    ttft = (first_token_time or end) - start
    decode_time = max(end - (first_token_time or start), 1e-9)
    print(f"[{n_new} tokens | prompt {input_ids.shape[1]} tokens | "
          f"first token {ttft:.2f}s | total {end - start:.2f}s | "
          f"throughput {n_new / (end - start):.1f} tok/s overall, "
          f"{max(n_new - 1, 0) / decode_time:.1f} tok/s decoding]")
    print()
