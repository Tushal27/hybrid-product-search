"""
WEEK 4, STEP 4: RLHF (Reinforcement Learning from Human Feedback) -- how
a model gets pushed toward outputs that score well on a REWARD signal,
instead of toward matching a fixed dataset like fine-tuning (step 2).

Fine-tuning has a target: "produce THIS exact text." RLHF doesn't -- it
has a SCORE: "make outputs that score higher on this reward." No single
correct completion is specified; the model can express the desired
quality however it wants, as long as reward goes up. Real RLHF trains a
whole separate REWARD MODEL on human preference comparisons ("response A
beats response B") to produce that score. That's its own substantial
topic; this demo hand-codes a simple, transparent reward function instead
(count positive-sentiment words minus negative ones) so the POLICY UPDATE
mechanism -- the actual "RL" part -- is what's on display.

THE ALGORITHM (a simplified REINFORCE, real RLHF usually uses PPO for
better stability, same core gradient idea): sample several completions,
score each with the reward function, and push up the log-probability of
tokens from ABOVE-average completions while pushing down tokens from
BELOW-average ones -- weighted by how far from average each one scored.

THE CRITICAL PIECE MOST SIMPLIFIED EXPLANATIONS SKIP: pure reward
maximization degenerates FAST -- the model discovers it can just repeat
"happy happy happy happy" forever and rack up huge reward, producing
garbage. Real RLHF adds a KL-divergence penalty that punishes the policy
for drifting too far from a FROZEN reference copy of the original model.
This script implements that penalty for real, and shows generations with
it in place -- reward-seeking, without degenerating into repetition spam.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import copy
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

torch.manual_seed(0)

print("=" * 78)
print("SETUP -- policy model (trained) + frozen reference copy (for the KL penalty)")
print("=" * 78)
tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token
policy_model = GPT2LMHeadModel.from_pretrained("distilgpt2")
reference_model = copy.deepcopy(policy_model)
for p in reference_model.parameters():
    p.requires_grad = False
reference_model.eval()
print("Two copies of distilgpt2: 'policy' (this is what gets trained) and 'reference'")
print("(frozen forever, exists only to measure how far the policy has drifted).")
print()

POSITIVE_WORDS = ["happy", "great", "wonderful", "love", "amazing", "joy", "excellent", "fantastic",
                  "good", "glad", "excited", "grateful", "fine", "hopeful", "proud", "relieved"]
NEGATIVE_WORDS = ["sad", "bad", "terrible", "hate", "awful", "horrible", "miserable",
                  "sick", "sorry", "worried", "tired", "anxious", "lonely", "upset", "afraid"]


def reward_fn(text):
    """Stand-in for a trained reward model -- hand-coded and transparent on
    purpose, so what's being demonstrated is the POLICY update, not reward
    model training (its own separate topic)."""
    t = text.lower()
    return sum(t.count(w) for w in POSITIVE_WORDS) - sum(t.count(w) for w in NEGATIVE_WORDS)


PROMPT = "Today I feel"
prompt_ids = tokenizer(PROMPT, return_tensors="pt").input_ids


def sample_continuation(model, prompt_ids, max_new_tokens=15):
    with torch.no_grad():
        return model.generate(prompt_ids, max_new_tokens=max_new_tokens, do_sample=True,
                               top_k=50, temperature=1.0, pad_token_id=tokenizer.eos_token_id)


def token_log_probs(model, full_ids, prompt_len):
    """log-prob the model assigns to each ALREADY-SAMPLED generated token --
    this is what both the policy gradient and the KL penalty are built from."""
    logits = model(full_ids).logits
    log_probs_all = F.log_softmax(logits, dim=-1)
    gen_ids = full_ids[0, prompt_len:]
    pred_positions = list(range(prompt_len - 1, full_ids.shape[1] - 1))
    return log_probs_all[0, pred_positions, :].gather(1, gen_ids.unsqueeze(1)).squeeze(1)


print(f"Prompt: {PROMPT!r}  -- sample completions BEFORE any RL training:")
for _ in range(3):
    seq = sample_continuation(policy_model, prompt_ids)
    text = tokenizer.decode(seq[0, prompt_ids.shape[1]:], skip_special_tokens=True)
    print(f"  {text!r}   reward={reward_fn(text)}")
print()


# ---------------------------------- the REINFORCE + KL-penalty loop ----------------------------------
print("=" * 78)
print("TRAINING -- REINFORCE with a baseline, plus a KL penalty against the reference")
print("=" * 78)
K = 6              # completions sampled per iteration
NUM_ITERS = 35
KL_COEF = 0.15
optimizer = torch.optim.Adam(policy_model.parameters(), lr=2e-5)

for it in range(NUM_ITERS):
    sequences, rewards = [], []
    for _ in range(K):
        seq = sample_continuation(policy_model, prompt_ids)
        text = tokenizer.decode(seq[0, prompt_ids.shape[1]:], skip_special_tokens=True)
        sequences.append(seq)
        rewards.append(reward_fn(text))
    rewards_t = torch.tensor(rewards, dtype=torch.float32)
    baseline = rewards_t.mean()               # variance-reduction trick: reward RELATIVE to this batch's average
    advantages = rewards_t - baseline

    policy_loss, kl_loss = 0.0, 0.0
    for k in range(K):
        logp_tokens = token_log_probs(policy_model, sequences[k], prompt_ids.shape[1])
        with torch.no_grad():
            ref_logp_tokens = token_log_probs(reference_model, sequences[k], prompt_ids.shape[1])
        # REINFORCE: push UP log-prob of tokens from above-average-reward completions,
        # push DOWN log-prob of tokens from below-average ones
        policy_loss = policy_loss + (-advantages[k] * logp_tokens.sum())
        # KL penalty: how far has the policy's probability of ITS OWN sampled tokens
        # drifted from what the frozen reference would have assigned them?
        kl_loss = kl_loss + (logp_tokens - ref_logp_tokens).sum()

    total_loss = policy_loss / K + KL_COEF * (kl_loss / K)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if it == 0 or (it + 1) % 5 == 0:
        print(f"  iter {it+1:>2}/{NUM_ITERS}   mean_reward={rewards_t.mean().item():+.2f}   "
              f"kl={kl_loss.item()/K:+.3f}   loss={total_loss.item():+.3f}")
print()

print("=" * 78)
print("AFTER TRAINING -- same prompt, reward-seeking WITHOUT degenerating")
print("=" * 78)
final_rewards = []
for _ in range(5):
    seq = sample_continuation(policy_model, prompt_ids)
    text = tokenizer.decode(seq[0, prompt_ids.shape[1]:], skip_special_tokens=True)
    final_rewards.append(reward_fn(text))
    print(f"  {text!r}   reward={reward_fn(text)}")
print()
avg_reward = sum(final_rewards) / len(final_rewards)
print(f"Average reward after training: {avg_reward:+.2f} (started near 0)")
print()
print("This moved toward positive-sentiment completions -- the reward signal alone,")
print("with no target text ever specified anywhere -- while the KL penalty kept the")
print("policy close enough to the reference model that it's still generating varied,")
print("legible English rather than degenerating into 'happy happy happy happy...'")
print("(try setting KL_COEF near 0 and rerunning to see that collapse happen for real")
print("-- it's the single most common RLHF failure mode, and now you've built the")
print("exact mechanism that both causes it and prevents it.)")
