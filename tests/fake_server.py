"""An in-process fake chat server for probe tests (httpx.MockTransport).

It counts one token per four characters and mimics Ollama's behavior on overflow:
a prompt longer than the window loses its front and keeps the last half window.
"""

from __future__ import annotations

import json
import re

import httpx

CODE_RE = re.compile(r"(ORCHID|LANTERN)-(\d{4})")


class FakeServer:
    def __init__(
        self,
        ctx: int = 4096,
        kind: str = "ollama",
        tools_ok: bool = True,
        tools_as_text: bool = False,
        think_leak: bool = False,
        honor_max_tokens: bool = True,
        reject_overflow: bool = False,
        parallel: bool = True,
        schema_ok: bool = True,
        weak_recall: bool = False,
    ) -> None:
        self.ctx = ctx
        self.kind = kind
        self.tools_ok = tools_ok
        self.tools_as_text = tools_as_text
        self.think_leak = think_leak
        self.honor = honor_max_tokens
        self.reject_overflow = reject_overflow
        self.parallel = parallel
        self.schema_ok = schema_ok
        self.weak_recall = weak_recall
        self.last_system: str | None = None
        self.requests = 0

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/api/version":
            return (
                httpx.Response(200, json={"version": "0.0-test"})
                if self.kind == "ollama"
                else httpx.Response(404)
            )
        if self.kind == "ollama":
            if path == "/api/tags":
                return httpx.Response(200, json={"models": [{"name": "fake:1b"}]})
            if path == "/api/ps":
                return httpx.Response(
                    200,
                    json={
                        "models": [
                            {"name": "fake:1b", "context_length": self.ctx, "size": 2 * 1024**3}
                        ]
                    },
                )
            if path == "/api/show":
                return httpx.Response(
                    200,
                    json={
                        "model_info": {
                            "general.architecture": "qwen3",
                            "qwen3.context_length": 40960,
                            "qwen3.block_count": 28,
                            "qwen3.attention.head_count": 16,
                            "qwen3.attention.head_count_kv": 8,
                            "qwen3.attention.key_length": 128,
                            "qwen3.attention.value_length": 128,
                        }
                    },
                )
        if path == "/props" and self.kind == "llama-server":
            return httpx.Response(
                200, json={"default_generation_settings": {"n_ctx": self.ctx}, "total_slots": 4}
            )
        if path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "fake:1b", "context_length": 40960}]})
        if path == "/v1/chat/completions":
            self.requests += 1
            return self.chat(json.loads(request.content))
        return httpx.Response(404, json={"error": "not found"})

    def chat(self, body: dict) -> httpx.Response:
        msgs = body["messages"]
        text = "\n".join(str(m.get("content") or "") for m in msgs)
        prompt_tokens = len(text) // 4 + 10
        visible = text
        if prompt_tokens > self.ctx:
            if self.reject_overflow:
                return httpx.Response(
                    400,
                    json={"error": {"message": "the request exceeds the available context size"}},
                )
            keep = self.ctx // 2
            visible = text[-keep * 4 :]
            prompt_tokens = keep
        system = next((m["content"] for m in msgs if m["role"] == "system"), None)
        cached = 0
        if system and system == self.last_system:
            cached = len(system) // 4
        self.last_system = system
        last = str(msgs[-1].get("content") or "")
        content, reasoning, calls, finish = "", "", [], "stop"
        max_tokens = body.get("max_tokens", 100)
        if "START CODE" in last:
            found = {w: d for w, d in CODE_RE.findall(visible)}
            if self.weak_recall:
                found.pop("ORCHID", None)
            content = " ".join(f"{w}-{d}" for w, d in found.items()) or "I do not know."
        elif msgs[-1]["role"] == "tool" or any(m["role"] == "tool" for m in msgs[-2:]):
            content = "It prints hello."
        elif body.get("tools") and ("Open " in last or "Read " in last):
            paths = re.findall(r"/tmp/llm-doctor-probe/\w+\.py", last)
            if self.tools_as_text:
                content = f'<tool_call>{{"name": "Read", "arguments": {{"file_path": "{paths[0]}"}}}}</tool_call>'
            elif not self.tools_ok:
                calls = [{"name": "read_file", "arguments": json.dumps({"path": paths[0]})}]
            else:
                use = paths if self.parallel else paths[:1]
                calls = [{"name": "Read", "arguments": json.dumps({"file_path": p})} for p in use]
            finish = "tool_calls" if calls else "stop"
        elif body.get("response_format"):
            data = {
                "city": "Vienna",
                "country": "Austria",
                "population_millions": 2.0,
                "is_capital": True,
            }
            if not self.schema_ok:
                data = {"name": "Vienna"}
            content = json.dumps(data)
        elif "17 + 25" in last:
            if self.think_leak:
                content = "<think>17 plus 25</think>42"
            else:
                reasoning, content = "17 plus 25 is 42", "42"
        elif "Count from 1" in last:
            n = max_tokens if self.honor else 120
            content = " ".join(str(i) for i in range(1, n + 1))
            finish = "length"
            completion = n
        else:
            content = "OK"
        completion = locals().get("completion", max(1, len(content) // 4))
        usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion,
            "prompt_tokens_details": {"cached_tokens": cached},
        }
        if body.get("stream"):
            return self.stream(content, reasoning, calls, finish, usage)
        msg = {"role": "assistant", "content": content}
        if reasoning:
            msg["reasoning"] = reasoning
        if calls:
            msg["tool_calls"] = [
                {"id": f"call_{i}", "type": "function", "function": c} for i, c in enumerate(calls)
            ]
        return httpx.Response(
            200, json={"choices": [{"message": msg, "finish_reason": finish}], "usage": usage}
        )

    def stream(self, content, reasoning, calls, finish, usage) -> httpx.Response:
        chunks = []
        if reasoning:
            chunks.append({"choices": [{"delta": {"reasoning": reasoning}}]})
        for i in range(0, len(content), 8):
            chunks.append({"choices": [{"delta": {"content": content[i : i + 8]}}]})
        for i, c in enumerate(calls):
            chunks.append(
                {
                    "choices": [
                        {
                            "delta": {
                                "tool_calls": [
                                    {
                                        "index": i,
                                        "id": f"call_{i}",
                                        "function": {"name": c["name"], "arguments": ""},
                                    }
                                ]
                            }
                        }
                    ]
                }
            )
            half = len(c["arguments"]) // 2
            for piece in (c["arguments"][:half], c["arguments"][half:]):
                chunks.append(
                    {
                        "choices": [
                            {
                                "delta": {
                                    "tool_calls": [{"index": i, "function": {"arguments": piece}}]
                                }
                            }
                        ]
                    }
                )
        chunks.append({"choices": [{"delta": {}, "finish_reason": finish}]})
        chunks.append({"choices": [], "usage": usage})
        body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
        return httpx.Response(
            200, content=body.encode(), headers={"content-type": "text/event-stream"}
        )
