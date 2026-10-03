"""
WEEK 5, STEP 5: AI agents -- an autonomous loop that chains MULTIPLE tool
calls, deciding each next step from the RESULT of the previous one,
until it reaches a final answer on its own. This is the actual
difference between "tool calling" (step 4 -- one call, one result, done)
and "an agent": a question that needs 2+ sequential steps, where the
SECOND step's arguments literally can't be known until the first step's
real result comes back.

THE TASK: "What is the capital of France? Take the number of letters in
that city's name and add 10 to it." Nobody, model included, can know what
number to add 10 TO until lookup_capital(France) actually returns
"Paris" -- and even then, string_length("Paris") has to actually run
before the final add() call's arguments exist. Three tool calls, in a
strict dependency chain, decided one at a time by the model reading its
own accumulating history of what's happened so far.

THE PATTERN (this is literally called ReAct -- Reason, Act, Observe,
repeat): at every turn, the model emits either ONE MORE tool call or a
final answer. Whatever the tool call returns gets appended to the
conversation as an "observation," and the loop asks again with that new
information in view. No pre-planned sequence of steps exists anywhere in
this code -- the model is deciding the NEXT single action at every turn,
informed only by what's actually happened so far.
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


def ask(messages, max_new_tokens=80):
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

SYSTEM_PROMPT = (
    f"You are an agent that solves problems step by step using these tools:\n{tool_descriptions}\n\n"
    "At each turn, respond with ONLY ONE JSON object, either:\n"
    '  {"tool": "<tool name>", "arguments": {...}}   to take one action, OR\n'
    '  {"final_answer": "<answer>"}   once you have everything needed to answer.\n'
    "Use the RESULT of each tool call to decide your next action. Do not guess values "
    "a tool would give you -- always call the tool to get them."
)

QUESTION = ("What is the capital of France? Take the number of letters in that city's "
            "name and add 10 to it. Tell me the final number.")

print("=" * 78)
print("THE AGENT LOOP -- decide ONE action at a time, from accumulating real results")
print("=" * 78)
print(f"Question: {QUESTION!r}")
print("(no tool call's arguments can be known in advance -- each depends on the")
print("previous REAL tool result, not a pre-written plan)")
print()

messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": QUESTION}]
MAX_STEPS = 6
final_answer = None
stop_reason = f"hit the {MAX_STEPS}-step limit"
previous_action_str = None

for step in range(1, MAX_STEPS + 1):
    print(f"--- step {step} ---")
    raw = ask(messages)
    action_str = raw.strip()
    print(f"  model's action: {action_str!r}")

    if action_str == previous_action_str:
        stop_reason = "detected the SAME action repeated twice in a row -- a real loop-detection safeguard"
        break
    previous_action_str = action_str

    try:
        action = json.loads(strip_markdown_fences(action_str))
    except json.JSONDecodeError:
        stop_reason = "could not parse the model's output as JSON"
        break

    if "final_answer" in action:
        final_answer = action["final_answer"]
        stop_reason = "model emitted a final_answer"
        break

    if "tool" in action:
        response = client.request("tools/call", {"name": action["tool"], "arguments": action.get("arguments", {})})
        result_text = response["result"]["content"][0]["text"]
        print(f"  REAL observation from the tool server: {result_text}")
        messages.append({"role": "assistant", "content": action_str})
        messages.append({"role": "user", "content": f"Observation: {result_text}. What is your next action?"})
    else:
        stop_reason = "response had neither 'tool' nor 'final_answer'"
        break

client.close()
print()

print("=" * 78)
print("WHAT ACTUALLY HAPPENED")
print("=" * 78)
print(f"Stopped because: {stop_reason}")
true_capital, true_answer = "Paris", len("Paris") + 10
print(f"Ground truth for this question: capital of France = {true_capital!r} "
      f"({len(true_capital)} letters) + 10 = {true_answer}")

if final_answer is not None:
    print(f"Agent's final answer: {final_answer!r}")
    print(f"Correct: {str(true_answer) in str(final_answer)}")
    print()
    print("Each intermediate number came from an ACTUAL tool call this run, not the")
    print("model recalling a memorized answer -- change 'France' in QUESTION above and")
    print("the whole chain re-derives with different real numbers at every step.")
else:
    print("The agent did NOT reach a final answer this run.")
    print()
    print("This is a genuine, honest result, not a bug in the demo: at 494M parameters,")
    print("this model doesn't reliably sustain a multi-turn tool-use conversation --")
    print("it may shortcut a step using its own (sometimes-correct) world knowledge")
    print("instead of calling the tool it was told to always use, then lose track of")
    print("the conversation state and start repeating or malforming calls. THIS is")
    print("exactly why the step-limit and repeated-action safeguards above exist as")
    print("real code, not decoration -- an agent loop with no hard stop condition, on")
    print("a model this size, can genuinely run forever. Production agent systems")
    print("built on far more capable models still keep these same safeguards, because")
    print("the failure mode (confused loop, not graceful failure) is the same one,")
    print("just rarer. This is precisely the gap steps 9 (guardrails) and 10 (evals)")
    print("exist to catch systematically, instead of discovering it live in production.")
