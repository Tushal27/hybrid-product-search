"""
WEEK 4, STEP 7: MCP (Model Context Protocol) -- how a model gets access
to tools, data, and external systems it wasn't trained with, without
those tools being hardcoded into the model itself.

Unlike every other Week 4 topic, this isn't about model WEIGHTS at all --
it's a PROTOCOL: a standard message format so ANY client (a chat app, an
IDE, an agent framework) can talk to ANY tool server (a database, a
calculator, a filesystem, an API wrapper) without the two ever having
been custom-built for each other. That's the actual point: before a
shared protocol, every app that wanted to give a model tool access had to
write its own bespoke integration for every tool. MCP standardizes the
three moves every integration needs:

  1. "what can you do?"     -- tools/list  (server describes its tools + their
                                            argument schemas, at runtime)
  2. "do this, with these arguments" -- tools/call (client asks, server executes,
                                            wherever the server actually lives)
  3. "here's what happened"  -- the result comes back as plain content the
                                            model can read and respond to

This script runs a REAL, SEPARATE process (_mcp_tool_server.py) as a
subprocess and exchanges genuine JSON-RPC 2.0 messages with it over
stdin/stdout -- the same wire format and method names real MCP uses.
Nothing here is simulated: the tool genuinely executes in a different
process than the one deciding to call it, which is the actual mechanism
that lets MCP tools live anywhere (your machine, a company server, a
third-party's infrastructure) independent of which model or app is using
them.

ONE HONEST STAND-IN: deciding WHICH tool to call and with what arguments
is normally the model's own job (a capable instruction/tool-use-trained
model reads the tool schemas and the user's question, then emits a
structured tool call). This demo uses a simple keyword rule for that one
decision instead -- distilgpt2 (this project's only local model) is a
base language model with no tool-calling training, so it can't reliably
do this part. Everything AROUND that one decision -- discovering tools,
sending the call, getting a real result back from a real separate
process -- is completely real, unmocked protocol mechanics.
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")
import os
import json
import subprocess

SERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_mcp_tool_server.py")


class MCPClient:
    def __init__(self, server_path):
        self.proc = subprocess.Popen(
            [sys.executable, server_path],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )
        self._next_id = 1

    def request(self, method, params=None):
        req = {"jsonrpc": "2.0", "id": self._next_id, "method": method, "params": params or {}}
        self._next_id += 1
        print(f"  --> client sends : {json.dumps(req)}")
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        response_line = self.proc.stdout.readline()
        response = json.loads(response_line)
        print(f"  <-- server replies: {json.dumps(response)}")
        return response

    def close(self):
        self.proc.stdin.close()
        self.proc.wait(timeout=5)


print("=" * 78)
print("1. LAUNCH THE TOOL SERVER AS A REAL, SEPARATE PROCESS")
print("=" * 78)
client = MCPClient(SERVER_PATH)
print(f"Spawned subprocess PID {client.proc.pid} running _mcp_tool_server.py")
print("(this process knows NOTHING about any model -- it just speaks JSON-RPC)")
print()

print("=" * 78)
print("2. INITIALIZE -- the standard MCP handshake")
print("=" * 78)
client.request("initialize")
print()

print("=" * 78)
print("3. tools/list -- discover what this server can do, AT RUNTIME")
print("=" * 78)
tools_response = client.request("tools/list")
tools = tools_response["result"]["tools"]
print(f"\nDiscovered {len(tools)} tools, with their full argument schemas -- a client")
print("never needs these hardcoded; it reads them fresh from the server every time:")
for tool in tools:
    print(f"  - {tool['name']}: {tool['description']}")
    print(f"    schema: {json.dumps(tool['inputSchema'])}")
print()

print("=" * 78)
print("4. A 'MODEL' DECIDES TO CALL A TOOL (keyword stand-in, see docstring),")
print("   THEN THE REAL SERVER PROCESS EXECUTES IT")
print("=" * 78)


def decide_tool_call(user_question):
    """Stand-in for what a real tool-use-trained model would do: read the
    question + the tool schemas above, and emit a structured call. This
    project's only local model (distilgpt2) has no tool-use training, so
    this is hardcoded logic instead -- explicitly NOT presented as the
    interesting part of this demo."""
    q = user_question.lower()
    if "capital" in q:
        for country in ["japan", "france", "spain", "italy", "canada"]:
            if country in q:
                return "lookup_capital", {"country": country}
    if "+" in q or "plus" in q:
        import re
        nums = [int(x) for x in re.findall(r"\d+", q)]
        if len(nums) == 2:
            return "add", {"a": nums[0], "b": nums[1]}
    return None, None


questions = ["What is the capital of Japan?", "What is 47 plus 89?"]
for question in questions:
    print(f"\nUser question: {question!r}")
    tool_name, arguments = decide_tool_call(question)
    print(f"'Model' decides to call: {tool_name}({arguments})")
    call_response = client.request("tools/call", {"name": tool_name, "arguments": arguments})
    result_text = call_response["result"]["content"][0]["text"]
    print(f"Real result, computed inside the SEPARATE server process: {result_text}")
    print(f"Final answer the model would give the user: {question.rstrip('?')} -> {result_text}")
print()

print("=" * 78)
print("5. WHAT AN UNKNOWN/MISUSED TOOL CALL LOOKS LIKE")
print("=" * 78)
bad_response = client.request("tools/call", {"name": "lookup_capital", "arguments": {"country": "Wakanda"}})
print(f"\nisError: {bad_response['result']['isError']}  -- the server reports failure as DATA")
print("in the response, not by crashing the connection -- the model gets the error")
print("text back and can react to it (apologize, try different arguments, etc.)")
print()

client.close()
print("=" * 78)
print("This is the entire mechanism: a shared message format so tool discovery,")
print("tool execution, and results all flow the same way regardless of which model")
print("or which tool is on the other end. What made this whole project's models")
print("interesting was always the WEIGHTS; MCP is deliberately the opposite -- a")
print("protocol that works identically whether the model behind it is a tiny toy")
print("transformer or a frontier model, because it never looks inside the model at")
print("all, only at the messages crossing the wire.")
