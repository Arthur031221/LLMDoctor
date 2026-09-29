"""Agent-readiness probes. Each probe returns one or more Check results.

All probes share a time budget. A probe that would start after the deadline is
skipped, and every request gets a timeout no longer than the time that is left.
"""

from __future__ import annotations

import json
import random
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from llm_doctor.endpoint.backend import BackendInfo
from llm_doctor.endpoint.client import ChatClient, ChatResult

PASS, WARN, FAIL, SKIP, INFO = "PASS", "WARN", "FAIL", "SKIP", "INFO"

PROBES = ("context", "tools", "json", "think", "max-tokens", "prefix-cache", "speed", "ram")
DEFAULT_LEVELS = (2048, 4096, 8192, 16384)
DEEP_LEVELS = (*DEFAULT_LEVELS, 32768)


@dataclass
class Check:
    id: str
    name: str
    status: str
    detail: str
    data: dict = field(default_factory=dict)
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
            "seconds": round(self.seconds, 2),
            "data": self.data,
        }


@dataclass
class ProbeState:
    client: ChatClient
    backend: BackendInfo
    deadline: float
    levels: tuple[int, ...] = DEFAULT_LEVELS
    # Fresh randomness per run, so a server's prompt cache never sees a repeated prompt.
    rng: random.Random = field(default_factory=random.Random)
    chars_per_token: float = 4.0
    overhead_tokens: int = 0
    usage_reported: bool = True
    thinks_anyway: bool = False
    effective_ctx: int | None = None
    truncated: bool = False
    checks: list[Check] = field(default_factory=list)
    on_check: Callable[[Check], None] | None = None

    def remaining(self) -> float:
        return self.deadline - time.monotonic()

    def timeout(self, cap: float = 45.0) -> float:
        return max(1.0, min(cap, self.remaining()))

    def add(self, check: Check) -> None:
        self.checks.append(check)
        if self.on_check:
            self.on_check(check)


# Filler text: plain sentences, numbers of at most three digits so they never look like
# the four-digit marker codes.
_SUBJECTS = [
    "The courier",
    "A gardener",
    "The archivist",
    "Our neighbor",
    "The pilot",
    "A clerk",
    "The baker",
    "An engineer",
    "The ferryman",
    "A student",
    "The mayor",
    "A painter",
]
_VERBS = [
    "moved",
    "counted",
    "painted",
    "repaired",
    "carried",
    "sorted",
    "measured",
    "labeled",
    "inspected",
    "cleaned",
    "stacked",
    "delivered",
]
_OBJECTS = [
    "the copper kettles",
    "twelve wooden crates",
    "the old lanterns",
    "a box of maps",
    "the garden chairs",
    "three brass keys",
    "the spare ropes",
    "a stack of letters",
    "the blue bicycles",
    "the tin whistles",
    "a basket of pears",
    "the ledger pages",
]
_PLACES = [
    "near the harbor",
    "behind the mill",
    "on the second floor",
    "by the river gate",
    "in the east wing",
    "under the bridge",
    "next to the bakery",
    "at the station",
]


def filler(rng: random.Random, n_chars: int) -> str:
    parts: list[str] = []
    size = 0
    while size < n_chars:
        s = (
            f"{rng.choice(_SUBJECTS)} {rng.choice(_VERBS)} {rng.choice(_OBJECTS)} "
            f"{rng.choice(_PLACES)} at {rng.randint(1, 12)} o'clock on day {rng.randint(1, 300)}. "
        )
        parts.append(s)
        size += len(s)
    return "".join(parts)


def _code(rng: random.Random, word: str) -> str:
    return f"{word}-{rng.randint(1000, 9999)}"


# Tool set modeled on a coding agent's core tools.
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read a file from the local filesystem and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Absolute path of the file"},
                    "offset": {"type": "integer", "description": "Line to start from"},
                    "limit": {"type": "integer", "description": "Number of lines to read"},
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Write",
            "description": "Write a file to the local filesystem, replacing it if it exists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["file_path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Edit",
            "description": "Replace an exact string in a file with a new string.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string"},
                    "old_string": {"type": "string"},
                    "new_string": {"type": "string"},
                    "replace_all": {"type": "boolean"},
                },
                "required": ["file_path", "old_string", "new_string"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": "Run a shell command and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "description": {"type": "string"},
                    "timeout": {"type": "integer"},
                },
                "required": ["command"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Grep",
            "description": "Search file contents with a regular expression.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "path": {"type": "string"},
                    "glob": {"type": "string"},
                    "output_mode": {
                        "type": "string",
                        "enum": ["content", "files_with_matches", "count"],
                    },
                },
                "required": ["pattern"],
                "additionalProperties": False,
            },
        },
    },
]
TOOL_SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}
AGENT_SYSTEM = (
    "You are a coding agent working in a repository. Act by calling the provided tools. "
    "Never describe a tool call in text, make the call."
)

_TYPES = {
    "string": str,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
    "object": dict,
    "array": list,
}


def validate(value, schema: dict, path: str = "$") -> list[str]:
    """Validate the JSON Schema subset used by the probes."""
    errors: list[str] = []
    t = schema.get("type")
    if t:
        py = _TYPES.get(t)
        ok = isinstance(value, py) if py else True
        if t in ("integer", "number") and isinstance(value, bool):
            ok = False
        if not ok:
            return [f"{path} should be {t}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path} must be one of {schema['enum']}")
    if isinstance(value, dict):
        props = schema.get("properties", {})
        for req in schema.get("required", []):
            if req not in value:
                errors.append(f"{path}.{req} is missing")
        for k, v in value.items():
            if k in props:
                errors += validate(v, props[k], f"{path}.{k}")
            elif schema.get("additionalProperties") is False:
                errors.append(f"{path}.{k} is not allowed")
    return errors


TEXT_TOOL_CALL = re.compile(
    r"<tool_call>|<function=|\"name\"\s*:\s*\"(Read|Write|Edit|Bash|Grep)\"|\bRead\("
)


def _slow(cid: str, name: str, r: ChatResult, t0: float) -> Check:
    """A request that ran out of time says nothing about the capability being probed."""
    return Check(
        cid,
        name,
        WARN,
        f"did not finish in time ({r.error}), raise --budget",
        seconds=time.monotonic() - t0,
    )


def _clip(s: str, n: int = 80) -> str:
    s = " ".join(s.split())
    return s if len(s) <= n else s[: n - 3] + "..."


def calibrate(st: ProbeState) -> Check | None:
    """Warm the model up and learn how many characters make a token on this server."""
    t0 = time.monotonic()
    base = st.client.chat(
        [{"role": "user", "content": "Reply with OK."}], max_tokens=64, timeout=st.timeout(60)
    )
    if base.timed_out:
        return Check(
            "connect",
            "connect",
            FAIL,
            f"first request {base.error}, the model may still be loading. Run again or raise --budget",
            seconds=time.monotonic() - t0,
        )
    if not base.ok:
        return Check(
            "connect",
            "connect",
            FAIL,
            base.error or "request failed",
            seconds=time.monotonic() - t0,
        )
    text = filler(st.rng, 4000)
    big = st.client.chat(
        [{"role": "user", "content": text + "\nReply with OK."}],
        max_tokens=64,
        timeout=st.timeout(60),
    )
    if (
        big.ok
        and base.prompt_tokens
        and big.prompt_tokens
        and big.prompt_tokens > base.prompt_tokens
    ):
        st.chars_per_token = len(text) / (big.prompt_tokens - base.prompt_tokens)
        st.overhead_tokens = base.prompt_tokens
    else:
        st.usage_reported = False
    st.thinks_anyway = bool(base.reasoning) or "<think>" in base.content
    return None


def probe_context(st: ProbeState) -> Check:
    t0 = time.monotonic()
    passed = 0
    rows = []
    recall_misses: list[int] = []
    status, detail = PASS, ""
    answer_tokens = 1024 if st.thinks_anyway else 48
    for level in st.levels:
        if st.remaining() < 5:
            rows.append({"level": level, "result": "skipped, time budget used"})
            break
        a, b = _code(st.rng, "ORCHID"), _code(st.rng, "LANTERN")
        target_tokens = int(level * 0.85) - st.overhead_tokens - 80
        body = filler(st.rng, int(max(200, target_tokens) * st.chars_per_token))
        msgs = [
            {
                "role": "system",
                "content": "The user message starts with a START CODE and ends with an END CODE. "
                "When asked, reply with both codes and nothing else.",
            },
            {
                "role": "user",
                "content": f"START CODE: {a}\n\n{body}\n\nEND CODE: {b}\n\n"
                "What are the START CODE and the END CODE? Reply as START=<code> END=<code>.",
            },
        ]
        expected = st.overhead_tokens + int(len(msgs[1]["content"]) / st.chars_per_token) + 40
        r = st.client.chat(msgs, max_tokens=answer_tokens, timeout=st.timeout(180))
        row: dict = {
            "level": level,
            "expected_prompt_tokens": expected,
            "prompt_tokens": r.prompt_tokens,
        }
        rows.append(row)
        if not r.ok:
            msg = r.error or "request failed"
            if r.status is None or r.timed_out:
                # Slow is not the same as truncated. Report what was verified and stop.
                row["result"] = "timeout"
                status = WARN
                detail = f"{level:,}-token level did not finish in time ({_clip(msg, 60)}), raise --budget"
                break
            row["result"] = "rejected"
            st.truncated = True
            status = FAIL
            detail = f"{level:,}-token prompt rejected: {_clip(msg, 100)}"
            break
        out = r.text()
        start_ok = a.split("-")[1] in out
        end_ok = b.split("-")[1] in out
        counted = bool(st.usage_reported and r.prompt_tokens)
        truncated = bool(counted and r.prompt_tokens < expected * 0.8)
        row.update(start_found=start_ok, end_found=end_ok, truncated=truncated)
        # The server's own token count is the primary signal. Missing codes only count as
        # truncation when the server does not report usage.
        lost = truncated or (not counted and start_ok != end_ok)
        if lost:
            st.truncated = True
            where = (
                "front"
                if end_ok and not start_ok
                else "back"
                if start_ok and not end_ok
                else "part"
            )
            kept = f", server kept {r.prompt_tokens:,} tokens" if truncated else ""
            status = FAIL
            detail = f"{where} of a {expected:,}-token prompt was dropped{kept}"
            row["result"] = f"{where} truncation"
            break
        if not (start_ok and end_ok):
            missed = "START" if end_ok else "END" if start_ok else "both"
            row["result"] = f"ok, model missed {missed} code"
            recall_misses.append(level)
            if not counted:
                continue
        row.setdefault("result", "ok")
        passed = level
    adv = st.backend.advertised_ctx
    loaded = st.backend.loaded_ctx
    # The largest level that came back intact. When the server reports the window it
    # allocated and that window sits between the last good and the first bad level,
    # it is the more precise number.
    st.effective_ctx = passed or None
    failed_level = next(
        (
            r["level"]
            for r in rows
            if "truncation" in str(r.get("result")) or r.get("result") == "rejected"
        ),
        None,
    )
    if loaded and passed and failed_level and passed <= loaded < failed_level:
        st.effective_ctx = loaded
    head = []
    if st.effective_ctx:
        head.append(f"{st.effective_ctx:,} effective")
    if adv:
        head.append(f"of {adv:,} advertised")
    if loaded and loaded != adv:
        head.append(f"(server loaded {loaded:,})")
    summary = " ".join(head)
    if status == PASS and recall_misses:
        status = WARN
        detail = (
            f"the server kept every token up to {passed:,}, but the model failed to repeat a marker "
            f"at {', '.join(f'{x:,}' for x in recall_misses)} tokens (weak recall, not truncation)"
        )
    elif status == PASS:
        top = st.levels[-1]
        if passed < top:
            status, detail = WARN, f"stopped at {passed:,} tokens, time budget used"
        else:
            detail = f"no truncation up to {top:,} tokens"
            if top < 32768:
                detail += ", run --deep to test 32k"
        if top >= 32768 and passed < 32768:
            status = WARN
    elif status == FAIL and st.effective_ctx and st.effective_ctx >= 32768:
        status = WARN
    detail = f"{summary}. {detail}" if summary else detail
    return Check(
        "context",
        "context",
        status,
        detail,
        {
            "levels": rows,
            "effective": st.effective_ctx,
            "advertised": adv,
            "loaded": loaded,
            "chars_per_token": round(st.chars_per_token, 2),
        },
        time.monotonic() - t0,
    )


def _judge_call(r: ChatResult, expect_name: str, expect_args: dict) -> tuple[str, str]:
    if not r.ok:
        if r.timed_out:
            return WARN, f"did not finish in time ({r.error})"
        return FAIL, r.error or "request failed"
    if not r.tool_calls:
        if TEXT_TOOL_CALL.search(r.content):
            return FAIL, f"tool call printed as text instead of tool_calls: {_clip(r.content)}"
        return FAIL, f"no tool call, replied: {_clip(r.content or r.reasoning or '(empty)')}"
    tc = r.tool_calls[0]
    if tc.name not in TOOL_SCHEMAS:
        return FAIL, f"called unknown tool {tc.name!r}"
    try:
        args = json.loads(tc.arguments or "{}")
    except ValueError:
        return FAIL, f"{tc.name} arguments are not valid JSON: {_clip(tc.arguments)}"
    errs = validate(args, TOOL_SCHEMAS[tc.name])
    if errs:
        return FAIL, f"{tc.name} arguments break the schema: {'; '.join(errs[:3])}"
    if tc.name != expect_name:
        return WARN, f"called {tc.name} instead of {expect_name}"
    for k, v in expect_args.items():
        if args.get(k) != v:
            return WARN, f"{tc.name}.{k} = {args.get(k)!r}, expected {v!r}"
    return PASS, f"{tc.name}({', '.join(f'{k}=...' for k in args)}) with valid arguments"


def probe_tools(st: ProbeState) -> list[Check]:
    checks: list[Check] = []
    path = "/tmp/llm-doctor-probe/app.py"
    msgs = [
        {"role": "system", "content": AGENT_SYSTEM},
        {"role": "user", "content": f"Open {path} so we can look at it."},
    ]
    t0 = time.monotonic()
    r = st.client.chat(msgs, max_tokens=512, tools=TOOLS, timeout=st.timeout(40))
    status, detail = _judge_call(r, "Read", {"file_path": path})
    checks.append(
        Check(
            "tool-call",
            "tool call",
            status,
            detail,
            {"tool_calls": [tc.__dict__ for tc in r.tool_calls]},
            time.monotonic() - t0,
        )
    )

    # Round trip: send the tool result back, the way an agent loop does.
    if r.ok and r.tool_calls and st.remaining() > 5:
        tc = r.tool_calls[0]
        t0 = time.monotonic()
        follow = [
            *msgs,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": tc.id or "call_0",
                        "type": "function",
                        "function": {"name": tc.name, "arguments": tc.arguments or "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": tc.id or "call_0",
                "name": tc.name,
                "content": "def main():\n    print('hello')\n",
            },
            {"role": "user", "content": "What does the file print? Answer in one short sentence."},
        ]
        r2 = st.client.chat(follow, max_tokens=256, tools=TOOLS, timeout=st.timeout(40))
        if not r2.ok:
            s2, d2 = FAIL, f"server rejected a tool result message: {_clip(r2.error or '', 120)}"
        elif "hello" in r2.text().lower() or r2.tool_calls:
            s2, d2 = PASS, "tool result accepted and used"
        else:
            s2, d2 = WARN, f"tool result accepted, answer ignored it: {_clip(r2.content)}"
        checks.append(Check("tool-result", "tool result", s2, d2, seconds=time.monotonic() - t0))
    else:
        checks.append(Check("tool-result", "tool result", SKIP, "needs a working tool call"))

    if st.remaining() > 5:
        t0 = time.monotonic()
        pmsgs = [
            {"role": "system", "content": AGENT_SYSTEM},
            {
                "role": "user",
                "content": "Read /tmp/llm-doctor-probe/a.py and /tmp/llm-doctor-probe/b.py. "
                "Call Read for both files in the same response.",
            },
        ]
        rp = st.client.chat(pmsgs, max_tokens=512, tools=TOOLS, timeout=st.timeout(40))
        names = [tc.name for tc in rp.tool_calls]
        if not rp.ok:
            sp, dp = FAIL, rp.error or "request failed"
        elif len(rp.tool_calls) >= 2 and all(n == "Read" for n in names[:2]):
            sp, dp = PASS, f"{len(rp.tool_calls)} calls in one response"
        elif len(rp.tool_calls) == 1:
            sp, dp = WARN, "one call per response, the agent will need extra round trips"
        else:
            sp, dp = _judge_call(rp, "Read", {})
            sp = FAIL if sp != PASS else WARN
        checks.append(
            Check(
                "tool-parallel", "parallel tools", sp, dp, {"calls": names}, time.monotonic() - t0
            )
        )
    else:
        checks.append(Check("tool-parallel", "parallel tools", SKIP, "time budget used"))

    if st.remaining() > 5:
        t0 = time.monotonic()
        rs = st.client.chat(msgs, max_tokens=512, tools=TOOLS, stream=True, timeout=st.timeout(40))
        ss, ds = _judge_call(rs, "Read", {"file_path": path})
        if rs.ok and rs.tool_calls and rs.tool_call_chunks == 0:
            ss, ds = WARN, "tool call arrived without tool_calls deltas"
        elif rs.ok and rs.tool_calls and ss == PASS:
            n = rs.tool_call_chunks
            ds = f"tool_calls arrived in {n} stream delta{'s' if n != 1 else ''}, arguments valid"
        elif rs.ok and not rs.tool_calls and r.tool_calls:
            ss = FAIL
            ds = "tool call works without streaming but not with stream=true: " + ds
        checks.append(
            Check(
                "tool-stream",
                "streaming tools",
                ss,
                ds,
                {"deltas": rs.tool_call_chunks},
                time.monotonic() - t0,
            )
        )
    else:
        checks.append(Check("tool-stream", "streaming tools", SKIP, "time budget used"))
    return checks


CITY_SCHEMA = {
    "type": "object",
    "properties": {
        "city": {"type": "string"},
        "country": {"type": "string"},
        "population_millions": {"type": "number"},
        "is_capital": {"type": "boolean"},
    },
    "required": ["city", "country", "population_millions", "is_capital"],
    "additionalProperties": False,
}


def probe_json(st: ProbeState) -> Check:
    t0 = time.monotonic()
    rf = {
        "type": "json_schema",
        "json_schema": {"name": "city_facts", "strict": True, "schema": CITY_SCHEMA},
    }
    r = st.client.chat(
        [{"role": "user", "content": "Give me facts about Vienna."}],
        max_tokens=400 if not st.thinks_anyway else 1500,
        response_format=rf,
        timeout=st.timeout(40),
    )
    if r.timed_out:
        return _slow("json-schema", "json schema", r, t0)
    if not r.ok:
        return Check(
            "json-schema",
            "json schema",
            FAIL,
            f"response_format json_schema failed: {_clip(r.error or '', 120)}",
            seconds=time.monotonic() - t0,
        )
    text = r.content.strip()
    try:
        value = json.loads(text)
    except ValueError:
        fenced = text.startswith("```")
        why = "wrapped in a code fence" if fenced else f"not JSON: {_clip(text)}"
        return Check(
            "json-schema", "json schema", FAIL, f"output {why}", seconds=time.monotonic() - t0
        )
    errs = validate(value, CITY_SCHEMA)
    if errs:
        return Check(
            "json-schema",
            "json schema",
            WARN,
            "valid JSON, schema not enforced: " + "; ".join(errs[:3]),
            seconds=time.monotonic() - t0,
        )
    return Check(
        "json-schema",
        "json schema",
        PASS,
        "output matches the schema",
        seconds=time.monotonic() - t0,
    )


def probe_think(st: ProbeState) -> Check:
    """Streamed, so even a cut-off answer shows where the reasoning went."""
    t0 = time.monotonic()
    r = st.client.chat(
        [{"role": "user", "content": "What is 17 + 25? Reply with just the number."}],
        max_tokens=600,
        thinking=None,
        stream=True,
        timeout=st.timeout(40),
    )

    def done(status: str, detail: str) -> Check:
        data = {"reasoning_field": r.reasoning_field, "finish_reason": r.finish_reason}
        return Check("think-tags", "think tags", status, detail, data, time.monotonic() - t0)

    if not r.ok and not r.timed_out:
        return done(FAIL, r.error or "request failed")
    if "<think>" in r.content or "</think>" in r.content:
        return done(FAIL, "reasoning leaks into content as <think> tags")
    if r.reasoning:
        note = "" if r.content.strip() else ", the answer did not arrive within the token limit"
        return done(PASS, f"reasoning returned separately in `{r.reasoning_field}`{note}")
    if not r.content.strip():
        return (
            _slow("think-tags", "think tags", r, t0) if r.timed_out else done(WARN, "empty answer")
        )
    cut = r.timed_out or r.finish_reason == "length"
    if cut and len(r.content) > 200:
        return done(
            WARN, "long untagged content was cut off, it may be reasoning without a <think> tag"
        )
    return done(PASS, "no reasoning text in content")


def probe_max_tokens(st: ProbeState) -> Check:
    t0 = time.monotonic()
    limit = 16
    r = st.client.chat(
        [{"role": "user", "content": "Count from 1 to 300, separated by spaces."}],
        max_tokens=limit,
        timeout=st.timeout(30),
    )
    if not r.ok:
        return Check(
            "max-tokens",
            "max_tokens",
            FAIL,
            r.error or "request failed",
            seconds=time.monotonic() - t0,
        )
    n = r.completion_tokens
    if n is None:
        n_est = len(re.findall(r"\d+", r.content))
        if n_est > limit * 2:
            return Check(
                "max-tokens",
                "max_tokens",
                FAIL,
                f"about {n_est} numbers generated with max_tokens={limit}",
                seconds=time.monotonic() - t0,
            )
        return Check(
            "max-tokens",
            "max_tokens",
            PASS,
            "output stopped early (usage not reported)",
            seconds=time.monotonic() - t0,
        )
    if n > limit + 2:
        return Check(
            "max-tokens",
            "max_tokens",
            FAIL,
            f"{n} tokens generated with max_tokens={limit}",
            {"completion_tokens": n},
            time.monotonic() - t0,
        )
    if r.finish_reason not in ("length", "max_tokens"):
        return Check(
            "max-tokens",
            "max_tokens",
            WARN,
            f"stopped at {n} tokens but finish_reason is {r.finish_reason!r}",
            seconds=time.monotonic() - t0,
        )
    return Check(
        "max-tokens",
        "max_tokens",
        PASS,
        f"stopped at {n} tokens, finish_reason length",
        seconds=time.monotonic() - t0,
    )


def probe_prefix_cache(st: ProbeState) -> Check:
    t0 = time.monotonic()
    ctx = st.effective_ctx or 4096
    tokens = min(4000, int(ctx * 0.6))
    prefix = "Reference notes for this session.\n" + filler(
        random.Random(st.rng.random()), int(tokens * st.chars_per_token)
    )

    def ask(q: str) -> ChatResult:
        return st.client.chat(
            [{"role": "system", "content": prefix}, {"role": "user", "content": q}],
            max_tokens=8,
            stream=True,
            timeout=st.timeout(40),
        )

    r1 = ask("Reply with the word one.")
    r2 = ask("Reply with the word two.") if r1.ok else r1
    if not (r1.ok and r2.ok) or r1.ttft is None or r2.ttft is None:
        return Check(
            "prefix-cache",
            "prefix cache",
            SKIP,
            (r2.error or r1.error or "no timing data"),
            seconds=time.monotonic() - t0,
        )
    data = {
        "ttft_first": round(r1.ttft, 3),
        "ttft_second": round(r2.ttft, 3),
        "cached_tokens": r2.cached_tokens,
        "prompt_tokens": r2.prompt_tokens,
    }
    speedup = r1.ttft / r2.ttft if r2.ttft > 0 else 0
    cached = r2.cached_tokens or 0
    hit = (r2.prompt_tokens and cached >= 0.5 * r2.prompt_tokens) or speedup >= 2
    detail = f"TTFT {r1.ttft:.2f} s then {r2.ttft:.2f} s on a repeated {r2.prompt_tokens or tokens:,}-token prefix"
    if r2.cached_tokens is not None:
        detail += f", {cached:,} tokens cached"
    return Check(
        "prefix-cache",
        "prefix cache",
        PASS if hit else WARN,
        detail if hit else detail + ", no reuse",
        data,
        time.monotonic() - t0,
    )


def probe_speed(st: ProbeState) -> list[Check]:
    out = []
    for size in (1024, 8192):
        cid = f"speed-{size // 1024}k"
        name = f"speed {size // 1024}k"
        if st.effective_ctx and st.effective_ctx < size + 512:
            out.append(
                Check(cid, name, SKIP, f"effective context {st.effective_ctx:,} is below {size:,}")
            )
            continue
        if st.remaining() < 8:
            out.append(Check(cid, name, SKIP, "time budget used"))
            continue
        t0 = time.monotonic()
        text = filler(
            random.Random(st.rng.random()),
            int((size - st.overhead_tokens - 60) * st.chars_per_token),
        )
        r = st.client.chat(
            [{"role": "user", "content": text + "\nSummarize the notes above in about 60 words."}],
            max_tokens=96,
            stream=True,
            timeout=st.timeout(60),
        )
        if r.timed_out and r.ttft is None:
            out.append(_slow(cid, name, r, t0))
            continue
        if not r.ok or r.ttft is None:
            out.append(
                Check(
                    cid, name, FAIL, r.error or "no tokens received", seconds=time.monotonic() - t0
                )
            )
            continue
        rate = r.decode_rate()
        pp = (r.prompt_tokens / r.ttft) if r.prompt_tokens and r.ttft else None
        parts = [f"TTFT {r.ttft:.2f} s"]
        if pp:
            parts.append(f"prefill {pp:,.0f} tok/s")
        if rate:
            parts.append(f"decode {rate:.1f} tok/s")
        status = INFO
        if (size >= 8192 and r.ttft > 30) or (rate is not None and rate < 5):
            status = WARN
        out.append(
            Check(
                cid,
                name,
                status,
                ", ".join(parts),
                {
                    "ttft": r.ttft,
                    "prefill_tps": pp,
                    "decode_tps": rate,
                    "prompt_tokens": r.prompt_tokens,
                },
                time.monotonic() - t0,
            )
        )
    return out
