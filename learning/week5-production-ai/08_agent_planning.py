"""
WEEK 5, STEP 8: Agent planning -- produce the WHOLE multi-step plan
UP FRONT, before executing anything, instead of step 5's ReAct pattern
(decide one action at a time, from whatever's happened so far).

Step 5 showed a real failure: a small model juggling "what's my overall
goal", "what's my next single action", AND "am I done yet" -- fresh, at
every single turn -- can lose the thread over a multi-turn conversation.
Plan-first changes the shape of the problem: the model makes ONE
decision (the whole plan, with placeholders for results it doesn't have
yet), and then ORDINARY CODE -- not another model call -- walks the plan
and fills in each placeholder with the REAL result of the step before it.
Fewer decision points for the model to get wrong.

This also unlocks something ReAct fundamentally can't: the plan can be
INSPECTED before a single tool actually runs. In ReAct, by the time you
see an action, it already happened. Here, a human (or a guardrail, step
9) could review and approve/edit the plan BEFORE any real-world side
effect occurs -- the actual reason production agent systems often prefer
some form of upfront planning for anything consequential (sending an
email, making a purchase, deleting a file), even though it's less
adaptive than pure ReAct if the world doesn't match the plan.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import json
import re
import subprocess
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.manual_seed(0)

SERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "..", "week4-modern-llms", "_mcp_tool_server.py")


class MCPClient:
    def __init__(self, server_path):
        self.proc = subprocess.Popen([sys.executable, server_path], stdin=subprocess.PIPE,
                                      stdout=subprocess.PIPE, text=True, bufsize=1)
        self._next_id = 1

    def request(self, method, params=None):
        req = {"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params or {}}
        self._next_id += 1
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        return json.loads(self.proc.stdout.readline())

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=5)


tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", dtype=torch.float32)
model.eval()


def ask(messages, max_new_tokens=150):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)


def strip_markdown_fences(text):
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1) if match else text


client = MCPClient(SERVER_PATH)
client.request("initialize")
tools = client.request("tools/list")["result"]["tools"]
tool_descriptions = "\n".join(f"- {t['name']}({json.dumps(t['inputSchema']['properties'])}): {t['description']}"
                               for t in tools)

EXAMPLE = json.dumps({"plan": [
    {"tool": "lookup_capital", "arguments": {"country": "Italy"}},
    {"tool": "string_length", "arguments": {"text": "RESULT_OF_STEP_1"}},
    {"tool": "add", "arguments": {"a": "RESULT_OF_STEP_2", "b": 5}},
]})

PLANNER_SYSTEM = (
    f"You have these tools:\n{tool_descriptions}\n\n"
    "Produce a JSON plan to answer the user's question: a list of steps, each "
    '{"tool": "<name>", "arguments": {...}}. If an argument depends on an EARLIER '
    'step\'s result, use the placeholder string "RESULT_OF_STEP_<n>" (1-indexed) '
    "instead of a real value. Respond with ONLY JSON: "
    f'{{"plan": [...]}}. Example, for "capital of Italy, city name length plus 5": {EXAMPLE}'
)

QUESTION = "What is the capital of Spain, and what is that city's name length plus 20?"

print("=" * 78)
print("1. PRODUCE THE WHOLE PLAN UP FRONT -- ONE model call, zero tools run yet")
print("=" * 78)
print(f"Question: {QUESTION!r}")
raw_plan = ask([{"role": "system", "content": PLANNER_SYSTEM}, {"role": "user", "content": QUESTION}])
print(f"Planner's raw output: {raw_plan.strip()!r}")

try:
    parsed = json.loads(strip_markdown_fences(raw_plan.strip()))
    plan = parsed["plan"]
except (json.JSONDecodeError, KeyError, TypeError):
    plan = []
print()
print("PLAN (inspectable BEFORE anything executes -- a human or guardrail could")
print("reject or edit this right here, something ReAct's step-at-a-time execution")
print("in week 5 step 5 structurally cannot offer):")
for i, step in enumerate(plan, 1):
    print(f"  {i}. {step['tool']}({step['arguments']})")
print()

# ---------------------------------- execute the plan -- CODE does the substitution, not the model ----------------------------------
print("=" * 78)
print("2. EXECUTE -- ordinary code substitutes each placeholder with the REAL prior result")
print("=" * 78)
step_results = {}
for i, step in enumerate(plan, 1):
    resolved_args = {}
    for key, value in step["arguments"].items():
        if isinstance(value, str) and value.startswith("RESULT_OF_STEP_"):
            ref = int(value.replace("RESULT_OF_STEP_", ""))
            resolved_args[key] = step_results[ref]
            print(f"  step {i}: substituting {value} -> {step_results[ref]!r} (the REAL step-{ref} result)")
        else:
            resolved_args[key] = value

    response = client.request("tools/call", {"name": step["tool"], "arguments": resolved_args})
    result_text = response["result"]["content"][0]["text"]
    try:
        result_value = int(result_text)  # numeric results get coerced so later add() steps can use them
    except ValueError:
        result_value = result_text
    step_results[i] = result_value
    print(f"  step {i}: {step['tool']}({resolved_args}) -> REAL result: {result_value!r}")
print()

client.close()

print("=" * 78)
print("3. RESULT")
print("=" * 78)
final_result = step_results.get(len(plan)) if plan else None
true_capital, true_answer = "Madrid", len("Madrid") + 20
print(f"Ground truth: capital of Spain = {true_capital!r} ({len(true_capital)} letters) + 20 = {true_answer}")
print(f"Plan's final result: {final_result}")
print(f"Correct: {final_result == true_answer}")
print()
print("Compare to step 5: there, the model had to correctly decide EVERY SINGLE turn,")
print("informed only by an accumulating, ever-longer conversation -- and it lost the")
print("thread by turn 3. Here, the model made exactly ONE structured decision (the")
print("plan), and deterministic code did the rest. Same underlying model, same tools,")
print("same class of question -- the RELIABILITY difference comes entirely from")
print("shrinking how much the model has to get right in a single pass.")
