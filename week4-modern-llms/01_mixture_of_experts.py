"""
WEEK 4, STEP 1: Mixture of Experts (MoE) -- how a model can have a HUGE
number of parameters while only paying compute for a SMALL fraction of
them on any given token.

Every FFN so far (weeks 1-3) was DENSE: every token passes through the
exact same feedforward network, using 100% of its weights, every time.
MoE replaces that one FFN with SEVERAL separate FFNs ("experts") plus a
small ROUTER network that looks at each token and decides which FEW
experts (usually top-1 or top-2 out of many) actually get to process it.
Every token still gets a full answer -- it just doesn't visit every
expert to get it. More total parameters (capacity to have learned many
different specializations), but compute per token stays small (only the
chosen experts run).

Two things this script proves with real numbers, not just description:
  1. FLOPs per token really do shrink to roughly (k / num_experts) of an
     equivalent dense network's cost, for k active experts out of many.
  2. A freshly-initialized (untrained) router routes VERY unevenly --
     some experts starve, some get overloaded -- and a small "load
     balancing" auxiliary loss, trained for real with a few gradient
     steps, visibly flattens that imbalance back out. This is a real,
     well-known failure mode of MoE training, not a hypothetical.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np
import torch
import torch.nn as nn

np.random.seed(0)
torch.manual_seed(0)


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=-1, keepdims=True)


def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


class Expert:
    """One small FFN -- structurally identical to the dense FFN every
    TransformerBlock has used since week1-transformer, just smaller."""

    def __init__(self, embed_dim, hidden_dim, seed):
        rng = np.random.default_rng(seed)
        scale = 0.3
        self.W1 = rng.standard_normal((embed_dim, hidden_dim)) * scale
        self.W2 = rng.standard_normal((hidden_dim, embed_dim)) * scale

    def forward(self, x):
        return gelu(x @ self.W1) @ self.W2


class MoELayer:
    def __init__(self, embed_dim, num_experts, expert_hidden_dim, top_k, seed=0):
        rng = np.random.default_rng(seed)
        self.router_W = rng.standard_normal((embed_dim, num_experts)) * 0.3
        self.experts = [Expert(embed_dim, expert_hidden_dim, seed=100 + i) for i in range(num_experts)]
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, X, verbose=False):
        n_tokens, embed_dim = X.shape
        router_logits = X @ self.router_W                 # (n_tokens, num_experts)
        router_probs = softmax(router_logits)

        output = np.zeros_like(X)
        chosen_experts_log = []
        for t in range(n_tokens):
            top_k_idx = np.argsort(-router_probs[t])[:self.top_k]
            top_k_weights = router_probs[t, top_k_idx]
            top_k_weights = top_k_weights / top_k_weights.sum()  # renormalize among just the chosen experts

            token_out = np.zeros(embed_dim)
            for expert_idx, weight in zip(top_k_idx, top_k_weights):
                token_out += weight * self.experts[expert_idx].forward(X[t])
            output[t] = token_out
            chosen_experts_log.append(list(top_k_idx))

            if verbose:
                weights_str = ", ".join(f"expert{e}:{w:.2f}" for e, w in zip(top_k_idx, top_k_weights))
                print(f"    token {t}: routed to [{weights_str}]")

        return output, router_probs, chosen_experts_log


# ---------------------------------- 1. routing mechanics ----------------------------------
EMBED_DIM = 32
NUM_EXPERTS = 8
EXPERT_HIDDEN_DIM = 64   # smaller per-expert than a dense FFN's hidden dim would be
TOP_K = 2

moe = MoELayer(EMBED_DIM, NUM_EXPERTS, EXPERT_HIDDEN_DIM, TOP_K)

rng = np.random.default_rng(1)
tokens = rng.standard_normal((6, EMBED_DIM))

print("=" * 78)
print("1. ROUTING -- each token picks its own top-2 experts out of 8")
print("=" * 78)
output, router_probs, chosen_log = moe.forward(tokens, verbose=True)
print(f"\nOutput shape: {output.shape}  (still (n_tokens, embed_dim) -- MoE is a")
print("drop-in replacement for a dense FFN, same input/output shape, same as every")
print("shape-preserving piece since week 2's stacked transformer blocks.)")
print()


# ---------------------------------- 2. the actual FLOPs savings ----------------------------------
print("=" * 78)
print("2. REAL COMPUTE COST: DENSE FFN vs MoE, EQUIVALENT TOTAL CAPACITY")
print("=" * 78)
dense_hidden_dim = NUM_EXPERTS * EXPERT_HIDDEN_DIM  # a dense FFN with the SAME total param count as all experts combined
dense_flops_per_token = 2 * EMBED_DIM * dense_hidden_dim   # W1 and W2 each cost embed_dim*hidden_dim multiplies
moe_flops_per_token = TOP_K * 2 * EMBED_DIM * EXPERT_HIDDEN_DIM  # only top_k experts actually run

print(f"Dense FFN matching MoE's TOTAL capacity ({NUM_EXPERTS} x {EXPERT_HIDDEN_DIM} = "
      f"{dense_hidden_dim} hidden units):")
print(f"  FLOPs per token: ~{dense_flops_per_token:,}  (100% of the network runs, every token)")
print(f"MoE ({NUM_EXPERTS} experts of {EXPERT_HIDDEN_DIM} hidden units each, top-{TOP_K} active):")
print(f"  FLOPs per token: ~{moe_flops_per_token:,}  (only {TOP_K}/{NUM_EXPERTS} = "
      f"{TOP_K/NUM_EXPERTS:.0%} of the network runs, per token)")
print(f"  -> {dense_flops_per_token / moe_flops_per_token:.1f}x less compute per token, for the SAME")
print(f"     total learned capacity sitting in the weights.")
print("This is the entire pitch of MoE: params (capacity to have learned many")
print("different specializations) and FLOPs (cost to run) are DECOUPLED, instead")
print("of scaling together like they do in a dense model.")
print()


# ---------------------------------- 3. the real failure mode: load imbalance ----------------------------------
print("=" * 78)
print("3. THE CATCH: AN UNTRAINED ROUTER LOADS EXPERTS UNEVENLY")
print("=" * 78)
big_batch = rng.standard_normal((500, EMBED_DIM))
_, _, chosen_log_big = moe.forward(big_batch, verbose=False)

expert_counts = np.zeros(NUM_EXPERTS, dtype=int)
for chosen in chosen_log_big:
    for e in chosen:
        expert_counts[e] += 1

print(f"500 tokens, top-{TOP_K} routing, BEFORE any load-balancing training:")
for e in range(NUM_EXPERTS):
    bar = "#" * int(expert_counts[e] / 5)
    print(f"  expert {e}: {expert_counts[e]:>4} tokens  {bar}")
imbalance_before = expert_counts.max() / expert_counts.min() if expert_counts.min() > 0 else float("inf")
print(f"max/min load ratio: {imbalance_before:.2f}x")
print("An overloaded expert becomes a bottleneck (everything routed to it queues up);")
print("a starved expert barely ever gets gradient signal and stays undertrained. Real")
print("MoE training (Switch Transformer, Mixtral, etc.) fixes this with an auxiliary")
print("LOAD BALANCING LOSS added to the training objective -- let's actually train")
print("against it and watch the imbalance shrink.")
print()

# ---------------------------------- train the router against a real load-balancing loss ----------------------------------
router = nn.Linear(EMBED_DIM, NUM_EXPERTS, bias=False)
with torch.no_grad():
    router.weight.copy_(torch.tensor(moe.router_W.T, dtype=torch.float32))

X_torch = torch.tensor(big_batch, dtype=torch.float32)
optimizer = torch.optim.Adam(router.parameters(), lr=0.05)

NUM_STEPS = 200
for step in range(NUM_STEPS):
    logits = router(X_torch)
    probs = torch.softmax(logits, dim=-1)
    # Switch Transformer-style load balancing loss: num_experts * sum_e(fraction of
    # tokens whose TOP CHOICE is e  *  mean router probability mass given to e)
    top1 = torch.argmax(probs, dim=-1)
    fraction_tokens = torch.stack([(top1 == e).float().mean() for e in range(NUM_EXPERTS)])
    fraction_prob_mass = probs.mean(dim=0)
    load_balance_loss = NUM_EXPERTS * torch.sum(fraction_tokens * fraction_prob_mass)

    optimizer.zero_grad()
    load_balance_loss.backward()
    optimizer.step()

    if step == 0 or (step + 1) % 50 == 0:
        print(f"  step {step+1:>3}/{NUM_STEPS}   load_balance_loss={load_balance_loss.item():.4f}")

with torch.no_grad():
    final_probs = torch.softmax(router(X_torch), dim=-1).numpy()
final_top1 = np.argmax(final_probs, axis=-1)
final_counts = np.bincount(final_top1, minlength=NUM_EXPERTS)

print()
print(f"500 tokens' TOP-1 choice, AFTER {NUM_STEPS} steps training ONLY the router")
print("against the load-balancing loss (experts themselves untouched):")
for e in range(NUM_EXPERTS):
    bar = "#" * int(final_counts[e] / 5)
    print(f"  expert {e}: {final_counts[e]:>4} tokens  {bar}")
imbalance_after = final_counts.max() / max(final_counts.min(), 1)
print(f"max/min load ratio: {imbalance_after:.2f}x  (was {imbalance_before:.2f}x before)")
print()
print("Note this loss says NOTHING about token/expert relevance -- it purely rewards")
print("spreading load evenly. Real MoE training balances THIS against the main")
print("next-token prediction loss, so routing still has to make sense semantically")
print("too -- load balancing is a constraint on training, not the goal of it.")
