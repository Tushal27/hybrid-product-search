"""
A minimal, REAL MCP-style tool server. Not a mock/simulation -- this is a
genuinely separate process, launched as a subprocess by 07_mcp.py, that
speaks newline-delimited JSON-RPC 2.0 over stdin/stdout, the same
transport real MCP servers use (stdio transport). It supports the same
three core methods a real MCP client relies on: initialize, tools/list,
tools/call.

Run standalone (feed it JSON-RPC requests on stdin, one per line) or let
07_mcp.py spawn and talk to it as a subprocess.
"""

import sys
import json

TOOLS = {
    "add": {
        "description": "Add two numbers together",
        "inputSchema": {
            "type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"],
        },
    },
    "lookup_capital": {
        "description": "Look up the capital city of a country",
        "inputSchema": {
            "type": "object",
            "properties": {"country": {"type": "string"}},
            "required": ["country"],
        },
    },
}

CAPITALS = {"japan": "Tokyo", "france": "Paris", "spain": "Madrid", "italy": "Rome", "canada": "Ottawa"}


def call_tool(name, arguments):
    if name == "add":
        return arguments["a"] + arguments["b"]
    if name == "lookup_capital":
        country = arguments["country"].lower()
        if country not in CAPITALS:
            raise ValueError(f"no capital on file for {arguments['country']!r}")
        return CAPITALS[country]
    raise ValueError(f"unknown tool: {name}")


def handle_request(req):
    method = req.get("method")
    req_id = req.get("id")

    if method == "initialize":
        result = {"protocolVersion": "2024-11-05", "serverInfo": {"name": "mini-mcp-server", "version": "0.1"}}
    elif method == "tools/list":
        result = {"tools": [{"name": name, **spec} for name, spec in TOOLS.items()]}
    elif method == "tools/call":
        params = req.get("params", {})
        try:
            value = call_tool(params["name"], params.get("arguments", {}))
            result = {"content": [{"type": "text", "text": str(value)}], "isError": False}
        except (KeyError, ValueError) as e:
            result = {"content": [{"type": "text", "text": str(e)}], "isError": True}
    else:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"unknown method {method!r}"}}

    return {"jsonrpc": "2.0", "id": req_id, "result": result}


if __name__ == "__main__":
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        request = json.loads(line)
        response = handle_request(request)
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
