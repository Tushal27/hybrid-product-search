"""
The assistant's core loop -- one user turn in, one guarded, tool-using,
memory-aware answer out.

DESIGN CHOICE, informed directly by week5-production-ai's results: keep
each turn to AT MOST one tool call, not an open-ended multi-step ReAct
loop. Week 5 step 5 showed a real, honest failure of a small (494M
param) model losing track across a multi-turn tool-use chain; step 8
showed a single upfront decision is far more reliable than several
sequential ones. Real personal-assistant requests ("what's 15% of 240",
"remember I like tea", "convert 5 miles to km") are almost always
single-step anyway -- so this scopes the agent to the pattern already
proven reliable, rather than the one already proven fragile. (A
plan-first extension, week5 step 8's pattern, is the natural next step
for genuinely compound requests -- deliberately left out of this first
version instead of risking the same instability step 5 hit.)

Tools execute IN-PROCESS here (a plain dispatch dict), not over a
subprocess/MCP transport like weeks 4-5's demos -- appropriate for a
single-user local CLI tool. A multi-user or networked version would
front this with the real MCP transport instead.
"""

import json
import re

import guardrails
import tools
from memory import Memory

FORGET_ALL_TOOL = "forget_all"  # deliberately outside guardrails.ALLOWED_TOOLS -- see below

# a general relevance gate, not just a prompt instruction -- testing showed this 494M
# model will hallucinate SOME tool call almost every time tools are in the system
# prompt, even when none apply (observed live: "What is the capital of Tokyo?" ->
# convert_units with invented numbers, and separately -> get_datetime, which needs no
# number at all, so a numeric-only check can't catch it). Each tool gets a keyword
# pattern that has to appear in the query for a call to that tool to even be considered
# plausible -- a real, deterministic check, not another instruction the model can ignore.
TOOL_RELEVANCE_PATTERNS = {
    "calculator": re.compile(r"[\d+\-*/%]|plus|minus|times|divided|multiplied|calculate|percent", re.I),
    "convert_units": re.compile(r"\d|convert|miles?\b|km\b|kilomet|meter|feet\b|foot\b|inch|pounds?\b|\blb\b|"
                                 r"ounces?\b|\bkg\b|grams?\b|celsius|fahrenheit|kelvin", re.I),
    "get_datetime": re.compile(r"\bdate\b|\btime\b|\btoday\b|\bnow\b|what day|\bclock\b", re.I),
    "remember_fact": re.compile(r"remember|note that|keep in mind|don'?t forget", re.I),
    FORGET_ALL_TOOL: re.compile(r"forget|delete|erase|clear (my|all) (memor)", re.I),
}

def _dispatch_remember(mem, args, original_query):
    # store the ORIGINAL utterance, not just the model's extracted "fact" argument --
    # a small model tends to over-extract ("teal" instead of "the user's favorite
    # color is teal"), and a bare keyword carries far too little context for the
    # mean-pooled-GloVe retrieval in memory.py to ever match it back to a later query
    mem.remember(original_query)
    return f"remembered: {original_query!r}"


TOOL_DISPATCH = {
    "calculator": lambda mem, args, query: tools.calculator(args["expression"]),
    "convert_units": lambda mem, args, query: tools.convert_units(args["value"], args["from_unit"], args["to_unit"]),
    "get_datetime": lambda mem, args, query: tools.get_datetime(),
    "remember_fact": _dispatch_remember,
    FORGET_ALL_TOOL: lambda mem, args, query: (mem.forget_all(), "all memories erased")[1],
}

ALL_SCHEMAS = dict(tools.TOOL_SCHEMAS)
ALL_SCHEMAS["remember_fact"] = {
    "description": "Store a fact about the user for later recall",
    "inputSchema": {"type": "object", "properties": {"fact": {"type": "string"}}, "required": ["fact"]},
}
ALL_SCHEMAS[FORGET_ALL_TOOL] = {
    "description": "Permanently erase ALL remembered facts (irreversible -- requires human confirmation)",
    "inputSchema": {"type": "object", "properties": {}, "required": []},
}


def _strip_markdown_fences(text):
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return m.group(1) if m else text


def _build_system_prompt(memory, query):
    tool_descriptions = "\n".join(
        f"- {name}({json.dumps(schema['inputSchema']['properties'])}): {schema['description']}"
        for name, schema in ALL_SCHEMAS.items()
    )
    relevant = memory.retrieve_relevant(query)
    memory_section = ""
    if relevant:
        facts = "\n".join(f"- {fact}" for _, fact in relevant)
        memory_section = f"\n\nRelevant facts you already know about the user:\n{facts}"

    return (
        f"You are a helpful personal assistant with access to these tools:\n{tool_descriptions}\n\n"
        "Only call a tool if one of these EXACT tools genuinely applies to the request. If nothing "
        "in this list truly fits (e.g. general knowledge questions, opinions, anything none of these "
        "tools do), just answer directly in plain text -- do NOT invent a tool call or force an "
        "unrelated tool just to produce JSON. "
        'When a tool DOES apply, respond with ONLY JSON: {"tool": "<name>", "arguments": {...}}. '
        "Only use remember_fact when the user is TELLING you something new to remember, never for a "
        "question -- if relevant facts are already shown below, just answer using them directly."
        f"{memory_section}"
    )


class AgentResult:
    def __init__(self, answer, tool_used=None, tool_result=None, blocked_reason=None,
                 pii_findings=None, trace=None):
        self.answer = answer
        self.tool_used = tool_used
        self.tool_result = tool_result
        self.blocked_reason = blocked_reason
        self.pii_findings = pii_findings or []
        self.trace = trace or []  # list of (stage_name, detail_string) -- every real step taken


def handle_turn(query, memory: Memory, ask_fn, confirm_fn=None):
    """ask_fn(messages) -> raw model text. confirm_fn() -> bool, asks a REAL human
    to confirm a destructive action; if None, destructive actions are always denied
    (safe default for non-interactive/demo use).

    Every AgentResult carries a `.trace` -- the literal sequence of real steps this
    turn took (guardrail checks, the exact system prompt built, the model's raw
    output, which tool ran and what it returned, the follow-up call). Nothing in
    trace is reconstructed after the fact; it's appended live as each step happens."""
    trace = []

    allowed, reason = guardrails.check_input(query)
    trace.append(("input guardrail", f"allowed={allowed}" + (f" ({reason})" if reason else "")))
    if not allowed:
        return AgentResult(answer="I can't process that request.", blocked_reason=reason, trace=trace)

    relevant = memory.retrieve_relevant(query)
    trace.append(("memory retrieval", f"{len(memory.facts)} facts stored, "
                                       f"{len(relevant)} cleared the relevance threshold: "
                                       f"{[f'{s:.2f} {f!r}' for s, f in relevant]}"))

    system_prompt = _build_system_prompt(memory, query)
    trace.append(("system prompt built", system_prompt))
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]

    raw = ask_fn(messages)
    trace.append(("model call 1 (raw output)", raw.strip()))

    try:
        call = json.loads(_strip_markdown_fences(raw.strip()))
    except json.JSONDecodeError:
        call = None
    trace.append(("parsed as tool call?", "yes: " + json.dumps(call) if isinstance(call, dict) and "tool" in call
                  else "no -- treated as a direct answer"))

    if not (isinstance(call, dict) and "tool" in call):
        clean, findings, redacted = guardrails.check_output(raw.strip())
        trace.append(("output guardrail", f"clean={clean}" + (f" findings={findings}" if findings else "")))
        return AgentResult(answer=redacted, pii_findings=findings, trace=trace)

    tool_name, arguments = call["tool"], call.get("arguments", {})

    def _force_direct_answer(safeguard_note, reask_note):
        trace.append(("safeguard", safeguard_note))
        followup = messages + [{"role": "user", "content": reask_note}]
        final_raw = ask_fn(followup)
        trace.append(("model call 2 (raw output)", final_raw.strip()))

        # the retry can still fail -- verified live (it repeated the exact same tool-call
        # JSON on the first retry attempt). Never let raw JSON reach the user either way;
        # fall back to an honest canned message instead of displaying `{"tool": ...}`.
        try:
            still_a_call = isinstance(json.loads(_strip_markdown_fences(final_raw.strip())), dict)
        except json.JSONDecodeError:
            still_a_call = False
        if still_a_call:
            trace.append(("safeguard", "retry STILL produced a tool call -- falling back to a canned answer "
                                        "rather than showing raw JSON to the user"))
            return AgentResult(answer="I don't have a tool that can answer that, and I'm having trouble "
                                       "phrasing a direct answer -- could you rephrase the question?",
                                pii_findings=[], trace=trace)

        clean, findings, redacted = guardrails.check_output(final_raw.strip())
        trace.append(("output guardrail", f"clean={clean}" + (f" findings={findings}" if findings else "")))
        return AgentResult(answer=redacted, pii_findings=findings, trace=trace)

    # a code-level safeguard, not just a prompt instruction: a question should never
    # get stored as a "remembered fact" -- catches exactly the failure mode observed
    # in testing (the model correctly recalled a fact from context, then ALSO
    # mis-fired remember_fact on the question itself, which would pollute memory
    # with questions instead of facts over time)
    if tool_name == "remember_fact" and query.strip().endswith("?"):
        return _force_direct_answer(
            "model tried to call remember_fact on a QUESTION -- overridden, re-asked for a direct answer",
            "Answer directly using the facts above, do not call remember_fact for a question.")

    # general relevance gate -- see TOOL_RELEVANCE_PATTERNS' comment. A prompt
    # instruction alone ("only call a tool if it genuinely applies") did not stop this
    # model from hallucinating SOME tool call almost every time, cycling through
    # different tools (convert_units with invented numbers, then get_datetime) on the
    # exact same irrelevant question depending only on what else was in context.
    pattern = TOOL_RELEVANCE_PATTERNS.get(tool_name)
    if pattern and not pattern.search(query):
        return _force_direct_answer(
            f"'{tool_name}' has no keyword signal in the query at all -- "
            f"the model invented arguments={arguments}. Treating as a hallucinated tool call.",
            "None of your tools apply to this request. Do NOT output JSON and do NOT call any tool -- "
            "write a plain English sentence answering the question directly.")

    if tool_name == FORGET_ALL_TOOL:
        confirmed = confirm_fn() if confirm_fn else False
        trace.append(("destructive-action gate", f"human confirmed={confirmed}"))
        if not confirmed:
            return AgentResult(answer="Forget-all was NOT performed -- it requires explicit human "
                                       "confirmation, which wasn't given.",
                                blocked_reason="destructive action not confirmed", trace=trace)
    elif not guardrails.check_tool_permission(tool_name):
        trace.append(("permission guardrail", f"'{tool_name}' NOT in allowed set -- blocked"))
        return AgentResult(answer=f"I can't use the '{tool_name}' tool.",
                            blocked_reason=f"'{tool_name}' is not in the permitted tool set", trace=trace)
    else:
        trace.append(("permission guardrail", f"'{tool_name}' allowed"))

    try:
        result = TOOL_DISPATCH[tool_name](memory, arguments, query)
    except (KeyError, ValueError, TypeError, ZeroDivisionError) as e:
        result = f"error: {e}"
    trace.append(("tool executed", f"{tool_name}({arguments}) -> {result!r}"))

    followup = messages + [
        {"role": "assistant", "content": raw.strip()},
        {"role": "user", "content": f"Tool result: {result}. Answer the original request in one "
                                     f"short sentence using this exact result."},
    ]
    final_raw = ask_fn(followup)
    trace.append(("model call 2 (raw output, with real tool result fed back)", final_raw.strip()))
    clean, findings, redacted = guardrails.check_output(final_raw.strip())
    trace.append(("output guardrail", f"clean={clean}" + (f" findings={findings}" if findings else "")))
    return AgentResult(answer=redacted, tool_used=tool_name, tool_result=result, pii_findings=findings, trace=trace)
