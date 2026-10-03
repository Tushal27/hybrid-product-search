"""
WEEK 5, STEP 2: Structured outputs -- getting a model to reliably produce
JSON that matches a specific schema, so downstream code can parse it
without guessing, instead of free-form prose you'd have to regex out
information from.

Three real production techniques stacked together:
  1. STATE THE SCHEMA EXPLICITLY in the prompt -- field names, types, and
     an instruction to emit ONLY JSON, nothing else.
  2. PARSE DEFENSIVELY -- models often wrap JSON in markdown code fences
     even when told not to; strip those before calling json.loads.
  3. VALIDATE AGAINST THE SCHEMA, THEN RETRY ON FAILURE -- if the parsed
     JSON is missing a required field or has the wrong type, send the
     model its own broken output plus the specific error and ask it to
     fix it, instead of just giving up on the first bad attempt.

No `jsonschema` library here -- a small hand-rolled validator, in keeping
with this project's whole approach: build the actual mechanism, not just
call something that hides it.
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


def ask(messages, max_new_tokens=80):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)


# ---------------------------------- a tiny hand-rolled schema validator ----------------------------------
SCHEMA = {
    "name": str,
    "age": int,
    "city": str,
}


def strip_markdown_fences(text):
    """Models frequently wrap JSON in ```json ... ``` even when told not to."""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1) if match else text


def validate(obj):
    if not isinstance(obj, dict):
        return "not a JSON object"
    for field, expected_type in SCHEMA.items():
        if field not in obj:
            return f"missing required field '{field}'"
        if not isinstance(obj[field], expected_type):
            return f"field '{field}' should be {expected_type.__name__}, got {type(obj[field]).__name__}"
    return None  # None means valid


def parse_and_validate(text):
    cleaned = strip_markdown_fences(text.strip())
    try:
        obj = json.loads(cleaned)
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"
    error = validate(obj)
    return (obj if error is None else None), error


SYSTEM_PROMPT = (
    'Extract information as JSON with EXACTLY these fields: '
    '{"name": <string>, "age": <integer>, "city": <string>}. '
    "Respond with ONLY the JSON object, no markdown, no explanation."
)

TEST_TEXTS = [
    "John Miller, 34, lives in Chicago and works as an engineer.",
    "Meet Priya Sharma -- she's 28 and just moved to Austin.",
    "Age 45. Name: Robert Chen. City: Seattle.",
]

print("=" * 78)
print("1. EXTRACT STRUCTURED DATA, VALIDATE AGAINST A SCHEMA")
print("=" * 78)
for text in TEST_TEXTS:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": text}]
    raw_response = ask(messages)
    obj, error = parse_and_validate(raw_response)
    print(f"Input : {text!r}")
    print(f"Raw   : {raw_response.strip()!r}")
    print(f"Result: {'VALID -> ' + json.dumps(obj) if error is None else 'INVALID -> ' + error}")
    print()


# ---------------------------------- the retry-on-failure pattern ----------------------------------
print("=" * 78)
print("2. RETRY-ON-INVALID -- force a failure, then recover from it")
print("=" * 78)
BAD_SYSTEM_PROMPT = "Tell me about this person."  # deliberately no schema/format instruction, to force a miss
tricky_text = "Age 45. Name: Robert Chen. City: Seattle."

messages = [{"role": "system", "content": BAD_SYSTEM_PROMPT}, {"role": "user", "content": tricky_text}]
attempt1 = ask(messages)
obj, error = parse_and_validate(attempt1)
print(f"Attempt 1 (deliberately unconstrained prompt): {attempt1.strip()[:80]!r}")
print(f"  -> {'VALID' if error is None else 'INVALID: ' + error}")

if error is not None:
    print("Retrying: feeding the model its OWN broken output + the specific error,")
    print("and re-asking with the schema this time...")
    retry_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": tricky_text},
        {"role": "assistant", "content": attempt1.strip()},
        {"role": "user", "content": f"That wasn't valid JSON matching the schema: {error}. "
                                     f"Respond again with ONLY the correct JSON object."},
    ]
    attempt2 = ask(retry_messages)
    obj2, error2 = parse_and_validate(attempt2)
    print(f"Attempt 2: {attempt2.strip()[:80]!r}")
    print(f"  -> {'VALID -> ' + json.dumps(obj2) if error2 is None else 'STILL INVALID: ' + error2}")
print()

print("This retry loop -- generate, validate, and if invalid feed the error back for")
print("one more try -- is a real, common production pattern for structured output:")
print("it doesn't require a bigger model, just code around the model that catches")
print("and corrects failures instead of assuming the first attempt always works.")
