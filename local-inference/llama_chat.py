"""
Chat with Qwen2.5 through llama.cpp -- the fast path on this machine.

Why this is faster than qwen_chat.py (PyTorch): llama.cpp runs a 4-bit GGUF
model (learning/week3-memory-performance/05_gguf.py) with kernels that work on an old
GTX 1050 Ti, via Vulkan. ~60 tok/s for the 1.5B model vs ~15 with PyTorch.

How it works: this script starts llama.cpp's own server (llama-server.exe) in
the background, then talks to it over HTTP with the OpenAI-style chat API and
prints tokens as they stream back. Same idea as qwen_chat.py: the conversation
lives in `messages` and is resent every turn -- but the server keeps a prompt
cache, so the part of the prompt it has already seen (the earlier turns) is
reused instead of recomputed (see the "cached" number in the stats line).

Setup (already done): llama-cpp/ holds the llama.cpp binaries + the GGUF model.

Usage:
    python local-inference/llama_chat.py
"""

import atexit
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

LLAMA_DIR = Path(__file__).resolve().parent.parent / "llama-cpp"   # binaries + model live in <repo>/llama-cpp (git-ignored)
MODEL_FILE = LLAMA_DIR / "models" / "qwen2.5-1.5b-instruct-q4_k_m.gguf"
PORT = 8089
CONTEXT = 4096          # tokens of prompt + reply the server keeps (VRAM is plentiful: model is ~1 GB)
MAX_NEW_TOKENS = 1024
TEMPERATURE = 0.7
TOP_P = 0.9
SYSTEM_PROMPT = (
    "You are a helpful assistant. You can see the whole conversation so far in the earlier messages. "
    "When the user refers to something said before (e.g. 'my previous question', 'what we discussed'), "
    "look back through those messages and answer from them."
)

BASE = f"http://127.0.0.1:{PORT}"


def start_server():
    """Launch llama-server (all layers on the GPU) and wait until the model is loaded."""
    print(f"Starting llama.cpp server with {MODEL_FILE.name}...")
    proc = subprocess.Popen(
        [str(LLAMA_DIR / "llama-server.exe"), "-m", str(MODEL_FILE), "-ngl", "99",
         "-c", str(CONTEXT), "--port", str(PORT), "--host", "127.0.0.1"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, cwd=LLAMA_DIR,
    )
    atexit.register(lambda: proc.poll() is None and proc.terminate())   # never leave it running
    for _ in range(120):
        if proc.poll() is not None:
            sys.exit("llama-server exited early -- try running llama-cpp\\llama-server.exe by hand to see why.")
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=1) as r:
                if json.load(r).get("status") == "ok":
                    return proc
        except Exception:
            pass
        time.sleep(0.5)
    proc.terminate()
    sys.exit("Timed out waiting for llama-server to load the model.")


def stream_chat(messages):
    """POST the conversation; yield ('text', chunk) for each piece, then ('stats', timings) at the end."""
    body = json.dumps({
        "messages": messages, "stream": True, "max_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE, "top_p": TOP_P,
        "cache_prompt": True,                       # reuse the KV cache of the unchanged start of the prompt
        "stream_options": {"include_usage": True},  # makes the last chunk carry token counts + timings
    }).encode()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        for raw in resp:                            # server-sent events: one "data: {...}" line per chunk
            line = raw.decode("utf-8").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event.get("choices"):
                piece = event["choices"][0]["delta"].get("content")
                if piece:
                    yield "text", piece
            if "timings" in event:
                yield "stats", event["timings"]


start_server()
messages = [{"role": "system", "content": SYSTEM_PROMPT}]   # the whole conversation, growing turn by turn
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
    print("Qwen: ", end="", flush=True)
    chunks, timings = [], {}
    start = time.perf_counter()
    first = None
    try:
        for kind, value in stream_chat(messages):
            if kind == "text":
                first = first or time.perf_counter()
                print(value, end="", flush=True)
                chunks.append(value)
            else:
                timings = value
    except Exception as e:
        print(f"\n[request failed: {e}]")
        messages.pop()   # forget the user message that got no answer
        continue
    total = time.perf_counter() - start

    messages.append({"role": "assistant", "content": "".join(chunks).strip()})
    new_prompt = timings.get("prompt_n", 0)       # prompt tokens actually processed this turn
    cached = timings.get("cache_n", 0)            # prompt tokens reused from the previous turn's cache
    print(f"\n[{timings.get('predicted_n', 0)} tokens | prompt {new_prompt + cached} tokens "
          f"({cached} reused from cache) | first token {(first or start) - start:.2f}s | "
          f"total {total:.2f}s | {timings.get('predicted_per_second', 0):.1f} tok/s generating, "
          f"{timings.get('prompt_per_second', 0):.0f} tok/s prompt]\n")
