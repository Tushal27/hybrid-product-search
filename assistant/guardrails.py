"""
Guardrails -- same three layers as week5-production-ai/09_guardrails.py,
now wired into a real application instead of a standalone demo.
"""

import re

INJECTION_PATTERNS = [
    r"ignore (all|any|the) (previous|prior|above) instructions",
    r"reveal (your|the) system prompt",
    r"you are now in developer mode",
]

PII_PATTERNS = {
    "SSN-like": r"\b\d{3}-\d{2}-\d{4}\b",
    "credit-card-like": r"\b\d{4}[- ]\d{4}[- ]\d{4}[- ]\d{4}\b",
}

# tools the agent may call WITHOUT extra confirmation. Destructive actions
# (forget_all) are deliberately NOT in here -- see agent.py's human-confirmation
# gate, a stronger guardrail than a simple allow-list for anything irreversible.
ALLOWED_TOOLS = {"calculator", "convert_units", "get_datetime", "remember_fact"}


def check_input(text):
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text.lower()):
            return False, f"blocked: matched injection pattern {pattern!r}"
    return True, None


def check_output(text):
    findings = [label for label, pattern in PII_PATTERNS.items() if re.search(pattern, text)]
    redacted = text
    for pattern in PII_PATTERNS.values():
        redacted = re.sub(pattern, "[REDACTED]", redacted)
    return (len(findings) == 0), findings, redacted


def check_tool_permission(tool_name):
    return tool_name in ALLOWED_TOOLS
