"""
Shared pieces for the English -> French translator: the base model, the
prompt format, a hand-written LoRA layer (same idea as learning/week4-modern-llms/
03_lora.py, but wrapped around the real Qwen model's nn.Linear layers), and
a batched translate() used by training, evaluation, and the web app.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import math
import os
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEVICE = os.environ.get("TRANSLATOR_DEVICE") or ("cuda" if torch.cuda.is_available() else "cpu")   # set TRANSLATOR_DEVICE=cpu to force CPU
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
ADAPTER_PATH = str(Path(__file__).parent / "lora_en_fr.pt")

SYSTEM_PROMPT = "Translate the user's English text into French. Reply with only the French translation."
LORA_RANK = 16
LORA_ALPHA = 32
TARGET_LAYERS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def load_base():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.padding_side = "left"   # batched generation needs the padding on the left
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=DTYPE).to(DEVICE).eval()
    return tokenizer, model


def build_prompt(tokenizer, english):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": english}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


class LoRALinear(nn.Module):
    """y = base(x) + (x @ A @ B) * (alpha / r).  base is frozen; only A and B train.
    B starts at zero so the model begins EXACTLY as the pretrained one."""

    def __init__(self, base: nn.Linear, r=LORA_RANK, alpha=LORA_ALPHA):
        super().__init__()
        self.base = base
        self.scale = alpha / r
        self.A = nn.Parameter(torch.empty(base.in_features, r, device=base.weight.device, dtype=torch.float32))
        self.B = nn.Parameter(torch.zeros(r, base.out_features, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + ((x.float() @ self.A) @ self.B * self.scale).to(x.dtype)

    @torch.no_grad()
    def merged_weight(self):
        """The patched weight W + scale * (A @ B)^T, as one ordinary matrix."""
        delta = (self.A @ self.B).T * self.scale
        return (self.base.weight.float() + delta).to(self.base.weight.dtype)


def add_lora(model):
    """Freeze the whole model, then wrap every attention/MLP linear layer in a LoRALinear."""
    for p in model.parameters():
        p.requires_grad_(False)
    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if name in TARGET_LAYERS and isinstance(child, nn.Linear):
                setattr(parent, name, LoRALinear(child))
    return model


def lora_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def load_translator(adapter_path=ADAPTER_PATH):
    """Base model + trained LoRA patch merged straight into the weights (zero extra inference cost)."""
    tokenizer, model = load_base()
    add_lora(model)
    state = torch.load(adapter_path, map_location=DEVICE)
    model.load_state_dict(state, strict=False)
    for module in list(model.modules()):
        for name, child in list(module.named_children()):
            if isinstance(child, LoRALinear):
                merged = child.base
                merged.weight.data = child.merged_weight()
                setattr(module, name, merged)
    return tokenizer, model.eval()


@torch.no_grad()
def translate(tokenizer, model, sentences, max_new_tokens=128, batch_size=16):
    """Greedy-decode French for each English string (batched, in the original order)."""
    results = []
    for i in range(0, len(sentences), batch_size):
        prompts = [build_prompt(tokenizer, s) for s in sentences[i:i + batch_size]]
        enc = tokenizer(prompts, return_tensors="pt", padding=True).to(DEVICE)
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
        for row in out:
            results.append(tokenizer.decode(row[enc.input_ids.shape[1]:], skip_special_tokens=True).strip())
    return results
