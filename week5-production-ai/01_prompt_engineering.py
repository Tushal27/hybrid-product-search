"""
WEEK 5, STEP 1: Prompt engineering -- the SAME model, the SAME question,
producing measurably more reliable answers purely from HOW the prompt is
written. This week uses a real instruction-tuned model for the first
time in the project (Qwen2.5-0.5B-Instruct, ~494M params, via
`transformers` -- distilgpt2 from week 4 is a base model with no
instruction-following training, so it can't reliably do any of this
week's topics).

THE TEST: sentiment classification. Downstream code needs to parse the
answer programmatically, so the ONLY acceptable outputs are the exact
words "positive", "negative", or "neutral" -- nothing else. Two prompting
strategies, same model, same 6 test reviews, same generation settings:

  VAGUE     : just ask the question, no format instruction, no examples.
  ENGINEERED: explicit role, explicit output constraint, few-shot examples
              showing the EXACT expected format.

Measuring two separate things per strategy: VALID (did the output parse
as exactly one of the three allowed words at all?) and CORRECT (did it
match the actual label?). A prompt can fail at either level independently
-- an eloquent, factually-right answer that isn't in the required format
is just as useless to downstream code as a wrong one.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)

print("=" * 78)
print("LOAD THE MODEL")
print("=" * 78)
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32)
model.eval()
print(f"{MODEL_NAME}: {sum(p.numel() for p in model.parameters()):,} params, instruction-tuned")
print()


def ask(messages, max_new_tokens=20):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)


TEST_REVIEWS = [
    ("This movie completely blew me away, I loved every minute.", "positive"),
    ("Waste of money, broke after two days.", "negative"),
    ("It arrived on time and does exactly what it says.", "neutral"),
    ("Absolutely fantastic service, will buy again!", "positive"),
    ("Terrible experience, the staff was rude and dismissive.", "negative"),
    ("The book is 300 pages long and covers three main topics.", "neutral"),
]

ALLOWED = {"positive", "negative", "neutral"}


def parse(response):
    cleaned = response.strip().lower().strip(".,!\"'")
    return cleaned if cleaned in ALLOWED else None


# ---------------------------------- strategy 1: vague ----------------------------------
print("=" * 78)
print("STRATEGY A -- VAGUE (no format instruction, no examples)")
print("=" * 78)
valid_a, correct_a = 0, 0
for text, label in TEST_REVIEWS:
    messages = [{"role": "user", "content": f"What do you think about this review: {text}"}]
    response = ask(messages, max_new_tokens=40)
    parsed = parse(response)
    is_valid = parsed is not None
    is_correct = parsed == label
    valid_a += is_valid
    correct_a += is_correct
    print(f"  review: {text[:45]:<45} -> {response.strip()[:60]!r}")
    print(f"    parsed={parsed}  valid={is_valid}  correct={is_correct}")
print(f"\nSTRATEGY A: {valid_a}/{len(TEST_REVIEWS)} valid format, {correct_a}/{len(TEST_REVIEWS)} correct")
print()

# ---------------------------------- strategy 2: engineered ----------------------------------
print("=" * 78)
print("STRATEGY B -- ENGINEERED (explicit role + constraint + few-shot examples)")
print("=" * 78)
SYSTEM_PROMPT = (
    "You are a sentiment classifier. Given a product/service review, respond with "
    "EXACTLY one word: positive, negative, or neutral. No punctuation, no explanation, "
    "no extra words -- just the single classification word."
)
FEW_SHOT = [
    {"role": "user", "content": "Review: I really enjoyed this, exceeded my expectations."},
    {"role": "assistant", "content": "positive"},
    {"role": "user", "content": "Review: Package arrived crushed and unusable."},
    {"role": "assistant", "content": "negative"},
    {"role": "user", "content": "Review: The manual has 12 pages and covers setup and maintenance."},
    {"role": "assistant", "content": "neutral"},
]

valid_b, correct_b = 0, 0
for text, label in TEST_REVIEWS:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}] + FEW_SHOT + \
               [{"role": "user", "content": f"Review: {text}"}]
    response = ask(messages, max_new_tokens=5)
    parsed = parse(response)
    is_valid = parsed is not None
    is_correct = parsed == label
    valid_b += is_valid
    correct_b += is_correct
    print(f"  review: {text[:45]:<45} -> {response.strip()[:60]!r}")
    print(f"    parsed={parsed}  valid={is_valid}  correct={is_correct}")
print(f"\nSTRATEGY B: {valid_b}/{len(TEST_REVIEWS)} valid format, {correct_b}/{len(TEST_REVIEWS)} correct")
print()

# ---------------------------------- comparison ----------------------------------
print("=" * 78)
print("RESULT")
print("=" * 78)
print(f"{'':<14} {'valid format':>14} {'correct':>10}")
print(f"{'VAGUE':<14} {f'{valid_a}/{len(TEST_REVIEWS)}':>14} {f'{correct_a}/{len(TEST_REVIEWS)}':>10}")
print(f"{'ENGINEERED':<14} {f'{valid_b}/{len(TEST_REVIEWS)}':>14} {f'{correct_b}/{len(TEST_REVIEWS)}':>10}")
print()
print("Same weights, same questions, same decoding settings (greedy, deterministic).")
print("The ONLY variable that changed is the prompt's structure: an explicit role,")
print("an explicit output constraint, and a few worked examples of the EXACT format")
print("expected. This is the entire craft of prompt engineering in one measurable")
print("comparison -- not 'magic words', but reducing the model's freedom to")
print("interpret what you actually wanted, in ways you can literally count.")
