from __future__ import annotations

import json

from fake_server import FakeServer

from llm_doctor.endpoint import run_endpoint
from llm_doctor.endpoint.client import split_base
from llm_doctor.endpoint.probes import TOOL_SCHEMAS, validate


def run(server: FakeServer, **kw):
    return run_endpoint("http://localhost:11434", transport=server.transport, budget=60, **kw)


def statuses(report) -> dict:
    return {c.id: c.status for c in report.checks}


def test_ollama_default_window_is_caught():
    report = run(FakeServer(ctx=4096))
    s = statuses(report)
    assert s["context"] == "FAIL"
    ctx = report.check("context")
    assert ctx.data["effective"] == 4096
    assert "front" in ctx.detail
    assert report.headline() == "Context: 4,096 effective of 40,960 advertised, FAIL"
    for cid in (
        "tool-call",
        "tool-result",
        "tool-parallel",
        "tool-stream",
        "json-schema",
        "think-tags",
        "max-tokens",
        "prefix-cache",
    ):
        assert s[cid] == "PASS", (cid, report.check(cid).detail)
    assert s["speed-8k"] == "SKIP"
    assert any("OLLAMA_CONTEXT_LENGTH=32768" in f for f in report.fixes)
    assert any("/api/chat accepts options.num_ctx" in f for f in report.fixes)
    json.dumps(report.to_dict())


def test_large_window_passes():
    report = run(FakeServer(ctx=65536))
    s = statuses(report)
    assert s["context"] == "PASS"
    assert report.check("context").data["effective"] == 16384
    assert s["speed-8k"] == "INFO"
    assert report.fixes == []


def test_broken_server_gets_fails_and_fixes():
    server = FakeServer(
        ctx=4096,
        kind="llama-server",
        tools_as_text=True,
        think_leak=True,
        honor_max_tokens=False,
        reject_overflow=True,
        schema_ok=False,
    )
    report = run(server)
    s = statuses(report)
    assert report.backend.kind == "llama-server"
    assert s["context"] == "FAIL" and "rejected" in report.check("context").detail
    assert s["tool-call"] == "FAIL" and "as text" in report.check("tool-call").detail
    assert s["think-tags"] == "FAIL"
    assert s["max-tokens"] == "FAIL"
    assert s["json-schema"] == "WARN"
    joined = " ".join(report.fixes)
    assert "-c 32768" in joined and "--reasoning-format deepseek" in joined and "--jinja" in joined


def test_hallucinated_tool_and_sequential_calls():
    report = run(FakeServer(ctx=65536, tools_ok=False))
    assert "unknown tool 'read_file'" in report.check("tool-call").detail
    report = run(FakeServer(ctx=65536, parallel=False), only={"tools"})
    assert statuses(report)["tool-parallel"] == "WARN"


def test_only_and_unreachable():
    report = run(FakeServer(), only={"max-tokens"})
    assert [c.id for c in report.checks] == ["max-tokens"]
    dead = run_endpoint("http://127.0.0.1:9", budget=10)
    assert dead.checks[0].status == "FAIL"


def test_helpers():
    assert split_base("http://h:8080/v1") == ("http://h:8080", "http://h:8080/v1")
    assert split_base("http://h:11434/") == ("http://h:11434", "http://h:11434/v1")
    assert split_base("http://h/v1/chat/completions") == ("http://h", "http://h/v1")
    assert validate({"file_path": "/x"}, TOOL_SCHEMAS["Read"]) == []
    assert validate({"path": "/x"}, TOOL_SCHEMAS["Read"]) == [
        "$.file_path is missing",
        "$.path is not allowed",
    ]
    assert validate({"pattern": "a", "output_mode": "all"}, TOOL_SCHEMAS["Grep"])
    assert validate({"command": 1}, TOOL_SCHEMAS["Bash"]) == ["$.command should be string, got int"]
