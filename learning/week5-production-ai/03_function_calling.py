"""
WEEK 5, STEP 3: Function calling -- the model outputs a structured
{"function": ..., "arguments": ...} object instead of prose, YOUR code
executes the real function, and the RESULT gets fed back so the model can
answer using real, computed information instead of guessing.

This is step 2's structured-output technique applied to one specific,
extremely common shape: a function name plus its arguments. The full
round trip has four parts:

  1. Tell the model what functions exist (name, description, argument schema)
  2. Model reads the user's question and emits a structured call, not an answer
  3. YOUR code parses that call and actually executes the real Python function
  4. The function's REAL result goes back to the model as a new message, and
     THEN the model writes the final natural-language answer

Notice the model never computes 47 + 89 itself -- it recognizes that this
question needs the add function, emits the call, and lets actual code do
the arithmetic. That's the entire point: offload the parts a language
model is unreliable at (precise math, real-time data, database lookups)
to real, deterministic code, while the model handles understanding the
request and communicating the result.
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


# ---------------------------------- the one real function the model can call ----------------------------------
def add(a, b):
    return a + b


FUNCTION_SCHEMA = {
    "name": "add",
    "description": "Add two numbers together and return the exact sum",
    "parameters": {"a": "number", "b": "number"},
}

SYSTEM_PROMPT = (
    f"You have access to this function: {json.dumps(FUNCTION_SCHEMA)}. "
    'If the user asks a question this function can answer, respond with ONLY a JSON '
    'object: {"function": "add", "arguments": {"a": <number>, "b": <number>}}. '
    "Otherwise, answer normally in plain text."
)


def strip_markdown_fences(text):
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1) if match else text


def try_parse_function_call(text):
    try:
        obj = json.loads(strip_markdown_fences(text.strip()))
        if isinstance(obj, dict) and "function" in obj and "arguments" in obj:
            return obj
    except json.JSONDecodeError:
        pass
    return None


def handle_question(question):
    print(f"User: {question!r}")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]

    step1 = ask(messages)
    print(f"  step 1 (model's raw output): {step1.strip()!r}")

    call = try_parse_function_call(step1)
    if call is None:
        print(f"  -> no function call detected, treating as direct answer: {step1.strip()!r}")
        return

    print(f"  step 2 (parsed function call): {call}")
    if call["function"] == "add":
        result = add(call["arguments"]["a"], call["arguments"]["b"])
    else:
        result = f"unknown function {call['function']!r}"
    print(f"  step 3 (REAL code executes 'add', not the model): {result}")

    followup_messages = messages + [
        {"role": "assistant", "content": step1.strip()},
        {"role": "user", "content": f"The function returned: {result}. Now answer the original "
                                     f"question in one short sentence using this exact result."},
    ]
    final_answer = ask(followup_messages, max_new_tokens=40)
    print(f"  step 4 (model's final answer, using the REAL computed result): {final_answer.strip()!r}")


print("=" * 78)
print("A QUESTION THE FUNCTION CAN ANSWER")
print("=" * 78)
handle_question("What is 4739 plus 8852?")
print()

print("=" * 78)
print("A QUESTION THE FUNCTION CANNOT ANSWER -- should NOT trigger a call")
print("=" * 78)
handle_question("What's a good name for a pet goldfish?")
print("(if this triggered an 'add' call anyway: that's a REAL, honest failure mode --")
print("a 494M model over-triggers tool calls on irrelevant questions sometimes. This")
print("is exactly the kind of thing step 9 (guardrails) and step 10 (evals) exist to")
print("catch before it reaches production, not something to paper over here.)")
print()

print("=" * 78)
print("VERIFY THE MODEL ISN'T JUST GUESSING THE ARITHMETIC ITSELF")
print("=" * 78)
big_numbers = (48317, 96284)
print(f"Asking the model to add {big_numbers[0]} + {big_numbers[1]} directly, WITHOUT the")
print("function available, to see whether it can actually do exact large-number")
print("arithmetic on its own (this is precisely the unreliability function calling")
print("exists to route around):")
direct = ask([{"role": "user", "content": f"What is {big_numbers[0]} + {big_numbers[1]}? Answer with just the number."}])
true_sum = sum(big_numbers)
print(f"  model's own answer: {direct.strip()!r}   actual answer: {true_sum}")
print(f"  model got it right on its own: {str(true_sum) in direct}")
