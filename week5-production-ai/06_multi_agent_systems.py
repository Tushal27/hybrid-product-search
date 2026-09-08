"""
WEEK 5, STEP 6: Multi-agent systems -- splitting a task across SEPARATE
model calls with DIFFERENT roles/system prompts, that pass structured
messages to each other, instead of one model call trying to do
everything at once.

Step 5's agent loop just showed a real limitation: a single small model
juggling "figure out the plan" AND "execute tool calls correctly" AND
"track multi-turn state" all at once can lose the thread. Multi-agent
splits those jobs apart, each with a narrower, easier-to-get-right job:

  PLANNER : reads the user's (possibly compound) question, breaks it into
            independent sub-questions -- reasoning ONLY, never touches a tool.
  EXECUTOR: answers ONE sub-question at a time using step 4's reliable
            single-tool-call pattern -- tool use ONLY, never sees the
            original compound question or has to plan anything.
  PLANNER : (again) receives all the sub-answers and synthesizes them into
            one final response -- back to reasoning ONLY.

Same underlying model (Qwen2.5-0.5B-Instruct) plays both roles here --
what makes them "different agents" is that each call gets its OWN
narrow system prompt and its OWN short-lived context, never the other
role's full history. That separation of concerns is the actual
architecture, independent of whether the roles happen to share weights
or not (real multi-agent systems sometimes use different models per
role specifically to put the more capable model on the harder job).
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

USER_QUESTION = "What is the capital of Japan, and separately, what is 123 plus 456?"

# ---------------------------------- agent 1: planner (decompose) ----------------------------------
print("=" * 78)
print("AGENT 1 (PLANNER) -- decompose the question, no tool access at all")
print("=" * 78)
PLANNER_SYSTEM = (
    "You are a planner. Break the user's question into a JSON list of independent, "
    'self-contained sub-questions. Respond with ONLY JSON: {"subquestions": ["...", "..."]}'
)
planner_response = ask([{"role": "system", "content": PLANNER_SYSTEM},
                         {"role": "user", "content": USER_QUESTION}])
print(f"User question: {USER_QUESTION!r}")
print(f"Planner's raw output: {planner_response.strip()!r}")

try:
    plan = json.loads(strip_markdown_fences(planner_response.strip()))
    subquestions = plan["subquestions"]
except (json.JSONDecodeError, KeyError):
    subquestions = [USER_QUESTION]  # fall back to treating it as one question
print(f"Parsed sub-questions: {subquestions}")
if any(sq[0].isupper() and sq.rstrip().endswith(".") for sq in subquestions):
    print("(honest note: the planner answered instead of just asking -- these read as")
    print("statements, not questions. A real system would validate/reject that. The")
    print("pipeline below still works because the executor is robust enough to pull")
    print("the right tool call out of a declarative sentence too -- worth noting, not")
    print("hiding: even a 'failed' planning step doesn't always break the whole chain.)")
print()

# ---------------------------------- agent 2: executor (one tool call each) ----------------------------------
print("=" * 78)
print("AGENT 2 (EXECUTOR) -- answers ONE sub-question at a time, tool access only,")
print("                      never sees the planner's reasoning or the other sub-Q's")
print("=" * 78)
EXECUTOR_SYSTEM = (
    f"You have access to these tools:\n{tool_descriptions}\n\n"
    'If a tool can answer the question, respond with ONLY JSON: {"tool": "<name>", "arguments": {...}}'
)

sub_answers = []
for sq in subquestions:
    print(f"Sub-question: {sq!r}")
    executor_response = ask([{"role": "system", "content": EXECUTOR_SYSTEM}, {"role": "user", "content": sq}])
    print(f"  executor's raw output: {executor_response.strip()!r}")
    try:
        call = json.loads(strip_markdown_fences(executor_response.strip()))
        tool_result = client.request("tools/call", {"name": call["tool"], "arguments": call["arguments"]})
        result_text = tool_result["result"]["content"][0]["text"]
        print(f"  REAL tool result: {result_text}")
    except (json.JSONDecodeError, KeyError):
        result_text = executor_response.strip()
        print(f"  (no valid tool call, using raw text as the answer)")
    sub_answers.append((sq, result_text))
    print()

client.close()

# ---------------------------------- agent 1 again: planner (synthesize) ----------------------------------
print("=" * 78)
print("AGENT 1 (PLANNER) AGAIN -- synthesize the sub-answers into one final response")
print("=" * 78)
synthesis_input = "\n".join(f"- {sq} -> {ans}" for sq, ans in sub_answers)
SYNTHESIZE_SYSTEM = "Combine these sub-answers into one clear, complete answer to the original question."
final_messages = [
    {"role": "system", "content": SYNTHESIZE_SYSTEM},
    {"role": "user", "content": f"Original question: {USER_QUESTION}\n\nSub-answers:\n{synthesis_input}"},
]
final_response = ask(final_messages, max_new_tokens=60)
print(f"Sub-answers gathered:\n{synthesis_input}")
print(f"\nFinal synthesized answer: {final_response.strip()!r}")
print()

print("=" * 78)
print("Three separate model calls, three narrow jobs, none of them needing to hold")
print("the WHOLE problem in view at once -- this is the actual value proposition of")
print("multi-agent systems: not 'more agents = smarter', but 'each agent's job is")
print("small enough to do reliably', which step 5 just demonstrated a single agent")
print("juggling everything at once can genuinely fail at.")
