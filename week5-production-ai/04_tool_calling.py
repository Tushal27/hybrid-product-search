"""
WEEK 5, STEP 4: Tool calling -- step 3's function-calling mechanism,
wired to MULTIPLE distinct real tools, with the model choosing WHICH one
fits a given question. This also closes a loop from week 4: the MCP demo
(week4-modern-llms/07_mcp.py) built a real subprocess tool server, but
used a keyword-matching stand-in for "the model decides which tool to
call" -- explicitly flagged there as a placeholder because distilgpt2
(week 4's only local model) has no instruction-following training to do
that job for real. This script has a real instruction-tuned model now
(Qwen2.5-0.5B-Instruct) -- so the stand-in gets replaced with the real
thing, over the SAME real MCP server subprocess from week 4.

Nothing here is simulated: the tool server is the actual
_mcp_tool_server.py subprocess from week4-modern-llms, speaking real
JSON-RPC 2.0 over stdin/stdout. The only thing that changed since week 4
is WHO decides which tool to call and with what arguments -- an actual
model reading the tool schemas at runtime, instead of a hardcoded rule.
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
    """Identical to week4-modern-llms/07_mcp.py's client -- same real subprocess,
    same real JSON-RPC 2.0 messages."""

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


def ask(messages, max_new_tokens=60):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                                     pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True)


def strip_markdown_fences(text):
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1) if match else text


print("=" * 78)
print("1. DISCOVER TOOLS FROM THE REAL MCP SERVER (same subprocess as week 4)")
print("=" * 78)
client = MCPClient(SERVER_PATH)
client.request("initialize")
tools = client.request("tools/list")["result"]["tools"]
for tool in tools:
    print(f"  - {tool['name']}: {tool['description']}  schema={json.dumps(tool['inputSchema'])}")
print()

# the system prompt is BUILT from the discovered schemas, not hand-typed for each tool
tool_descriptions = "\n".join(f"- {t['name']}({json.dumps(t['inputSchema']['properties'])}): {t['description']}"
                               for t in tools)
SYSTEM_PROMPT = (
    f"You have access to these tools:\n{tool_descriptions}\n\n"
    'If a tool can answer the user, respond with ONLY JSON: '
    '{"tool": "<tool name>", "arguments": {...}}. Otherwise answer normally.'
)


def handle_question(question):
    print(f"User: {question!r}")
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}]
    raw = ask(messages)
    print(f"  model's raw output: {raw.strip()!r}")

    try:
        call = json.loads(strip_markdown_fences(raw.strip()))
    except json.JSONDecodeError:
        call = None

    if not isinstance(call, dict) or "tool" not in call:
        print(f"  -> no tool call detected, direct answer: {raw.strip()!r}")
        return

    print(f"  model chose tool: {call['tool']}({call['arguments']})")
    response = client.request("tools/call", {"name": call["tool"], "arguments": call["arguments"]})
    result_text = response["result"]["content"][0]["text"]
    print(f"  REAL result from the separate MCP server process: {result_text}")

    followup = messages + [
        {"role": "assistant", "content": raw.strip()},
        {"role": "user", "content": f"The tool returned: {result_text}. Answer the original "
                                     f"question in one short sentence using this exact result."},
    ]
    final = ask(followup, max_new_tokens=40)
    print(f"  final answer: {final.strip()!r}")


print("=" * 78)
print("2. THE MODEL PICKS THE RIGHT TOOL AMONG SEVERAL, PER QUESTION")
print("=" * 78)
handle_question("What is the capital of France?")
print()
handle_question("What is 356 plus 789?")
print()
handle_question("What's the weather like today?")  # neither tool applies -- should NOT force a call
print("(another honest limitation, same category as step 3: a 494M model can pick the")
print("WRONG tool for a question neither tool actually answers. Notice it didn't crash")
print("though -- the tool failure came back as DATA (week 4's 'errors as data' point),")
print("and the model's final answer, while wrong, is a coherent reaction to that error")
print("rather than a broken pipeline. Real systems add tool-relevance checks and")
print("evals -- steps 9 and 10 -- specifically to catch this before a user sees it.)")
print()

client.close()
print("=" * 78)
print("This is the actual difference between 'function calling' and 'tool calling':")
print("function calling (step 3) is the MECHANISM -- structured call in, real result")
print("out. Tool calling is that same mechanism applied to a live ecosystem of")
print("MULTIPLE tools the model has to correctly choose AMONG, executing wherever")
print("those tools actually live -- here, a genuinely separate MCP server process,")
print("exactly like a real production agent would call a search API, a database,")
print("and a calculator, all through the same uniform interface.")
