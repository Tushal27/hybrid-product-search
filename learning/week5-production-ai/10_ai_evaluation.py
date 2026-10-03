"""
WEEK 5, STEP 10: AI evaluation (evals) -- the formal version of something
this week already did informally back in step 1: define a test SUITE with
known-good answers, run the model against every case, and get a SCORE,
so "is this prompt/model/config actually better" is a measured number,
not a vibe from eyeballing a few examples.

Three real eval patterns:
  1. EXACT-MATCH / RULE-BASED: objective tasks where correctness is
     checkable by code (a fact, a format, a computed value) -- fast, free,
     completely reliable, but only works when "correct" can be written as
     a rule.
  2. LLM-AS-JUDGE: for open-ended output with no single right string
     (tone, helpfulness, style), use a SECOND model call to score the
     first one's output against a rubric. Less reliable than exact-match,
     but covers what exact-match structurally cannot.
  3. REGRESSION GATING: run the SAME eval suite against two different
     configurations (here: step 1's vague vs. engineered prompt) and
     require a SCORE THRESHOLD before calling a change "safe to ship" --
     this is the actual mechanism behind "we added eval gates to CI" in
     any real LLM-backed product.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32)
model.eval()


def ask(messages, max_new_tokens=40):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)


def strip_markdown_fences(text):
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return m.group(1) if m else text


# ==================================== 1. EXACT-MATCH EVAL SUITE ====================================
print("=" * 78)
print("1. EXACT-MATCH EVAL SUITE -- objective, checkable-by-code cases")
print("=" * 78)
EVAL_SUITE = [
    ("What is the capital of Japan? Answer with just the city name.", "tokyo"),
    ("What is 12 plus 15? Answer with just the number.", "27"),
    ("What is the capital of France? Answer with just the city name.", "paris"),
    ("What is 9 times 6? Answer with just the number.", "54"),
]


def run_eval_suite(system_prompt=None):
    passed = 0
    for question, expected in EVAL_SUITE:
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + \
                   [{"role": "user", "content": question}]
        response = ask(messages, max_new_tokens=15)
        is_pass = expected in response.lower()
        passed += is_pass
        print(f"  Q: {question:<55} expected~={expected!r:<8} got={response.strip()[:30]!r:<32} "
              f"{'PASS' if is_pass else 'FAIL'}")
    return passed / len(EVAL_SUITE)


score = run_eval_suite()
print(f"\nScore: {score:.0%} ({int(score*len(EVAL_SUITE))}/{len(EVAL_SUITE)})")
print()


# ==================================== 2. LLM-AS-JUDGE ====================================
print("=" * 78)
print("2. LLM-AS-JUDGE -- scoring open-ended output that has no single correct string")
print("=" * 78)
OPEN_ENDED_PROMPT = "Explain why the sky is blue, in one friendly sentence for a curious 8-year-old."
response = ask([{"role": "user", "content": OPEN_ENDED_PROMPT}], max_new_tokens=60)
print(f"Model's response: {response.strip()!r}")

JUDGE_SYSTEM = (
    "You are an evaluator. Given a question and a response, judge it against this "
    'rubric and respond with ONLY JSON: {"accurate": true/false, "age_appropriate": true/false, '
    '"friendly_tone": true/false}'
)
judge_messages = [
    {"role": "system", "content": JUDGE_SYSTEM},
    {"role": "user", "content": f"Question: {OPEN_ENDED_PROMPT}\nResponse: {response.strip()}"},
]
verdict_raw = ask(judge_messages, max_new_tokens=40)
print(f"Judge's raw verdict: {verdict_raw.strip()!r}")
try:
    verdict = json.loads(strip_markdown_fences(verdict_raw.strip()))
    print(f"Parsed verdict: {verdict}")
    print(f"All criteria passed: {all(verdict.values())}")
except json.JSONDecodeError:
    print("(judge's output didn't parse as JSON this run -- a real system would retry")
    print("or fall back to a human review here, same defensive pattern as step 2)")
print()
print("Same model playing two roles again (step 6's multi-agent pattern) -- but this")
print("is a real limitation to be honest about: a judge only as good as the model")
print("judging, and a small model judging itself has an obvious blind spot (it may")
print("rate its own mistakes as fine). Production systems often use a STRONGER model")
print("as judge than the one being evaluated, specifically to reduce this bias.")
print()


# ==================================== 3. REGRESSION GATING ====================================
print("=" * 78)
print("3. REGRESSION GATING -- same eval suite, two prompt CONFIGS, a hard threshold")
print("=" * 78)
print("(the facts/arithmetic suite above is too easy to tell configs apart on -- both")
print("score 100% regardless of prompt. Reusing step 1's sentiment-format eval instead,")
print("since format compliance is exactly the kind of thing that DOES swing hugely")
print("with prompt changes -- a more honest test of what a regression gate is for.)")
print()

FORMAT_EVAL_SUITE = [
    ("This movie completely blew me away, I loved every minute.", "positive"),
    ("Waste of money, broke after two days.", "negative"),
    ("It arrived on time and does exactly what it says.", "neutral"),
    ("Absolutely fantastic service, will buy again!", "positive"),
]
ALLOWED = {"positive", "negative", "neutral"}


def run_format_eval_vague():
    """Exactly step 1's VAGUE strategy -- no system prompt, no format constraint."""
    valid = 0
    for text, _ in FORMAT_EVAL_SUITE:
        messages = [{"role": "user", "content": f"What do you think about this review: {text}"}]
        response = ask(messages, max_new_tokens=40).strip().lower().strip(".,!\"'")
        valid += response in ALLOWED
    return valid / len(FORMAT_EVAL_SUITE)


def run_format_eval_engineered():
    """Exactly step 1's ENGINEERED strategy -- explicit constraint, no few-shot needed
    to make the point here."""
    system_prompt = ("Respond with EXACTLY one word: positive, negative, or neutral. "
                      "No punctuation, no explanation.")
    valid = 0
    for text, _ in FORMAT_EVAL_SUITE:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": f"Review: {text}"}]
        response = ask(messages, max_new_tokens=8).strip().lower().strip(".,!\"'")
        valid += response in ALLOWED
    return valid / len(FORMAT_EVAL_SUITE)


REQUIRED_SCORE = 0.75
score_a = run_format_eval_vague()
score_b = run_format_eval_engineered()
print(f"Config A (vague, no format instruction): {score_a:.0%} valid format")
print(f"Config B (explicit format constraint)  : {score_b:.0%} valid format")
print()
print(f"Required threshold to ship: {REQUIRED_SCORE:.0%}")
print(f"Config A: {'PASSES' if score_a >= REQUIRED_SCORE else 'FAILS'} the gate ({score_a:.0%})")
print(f"Config B: {'PASSES' if score_b >= REQUIRED_SCORE else 'FAILS'} the gate ({score_b:.0%})")
print()
print("This is the entire point of building an eval suite once: any future prompt")
print("edit, model swap, or fine-tune (weeks 2-4) gets checked against the SAME fixed")
print("test cases before it ships, instead of 'it looked fine in the two examples I")
print("tried' -- exactly the gap that turns a demo into something safe to actually")
print("put in front of users.")
