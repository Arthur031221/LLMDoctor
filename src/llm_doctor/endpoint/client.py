"""Minimal chat client for OpenAI-compatible servers and Ollama's native /api/chat."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class ToolCall:
    name: str
    arguments: str  # raw JSON text as the server sent it
    id: str | None = None


@dataclass
class ChatResult:
    ok: bool
    status: int | None = None
    error: str | None = None
    content: str = ""
    reasoning: str = ""
    reasoning_field: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    ttft: float | None = None
    elapsed: float = 0.0
    first_token_at: float | None = None
    last_token_at: float | None = None
    tool_call_chunks: int = 0
    content_chunks: int = 0

    def text(self) -> str:
        return f"{self.reasoning}\n{self.content}"

    def decode_rate(self) -> float | None:
        if not self.completion_tokens or self.first_token_at is None or self.last_token_at is None:
            return None
        span = self.last_token_at - self.first_token_at
        if span <= 0 or self.completion_tokens < 2:
            return None
        return (self.completion_tokens - 1) / span


def split_base(url: str) -> tuple[str, str]:
    """Return (server root, OpenAI base ending in /v1 or similar)."""
    u = url.rstrip("/")
    for suffix in ("/chat/completions", "/v1/chat/completions"):
        if u.endswith(suffix):
            u = u[: -len(suffix)]
    if u.endswith("/v1"):
        return u[: -len("/v1")], u
    return u, u + "/v1"


class ChatClient:
    def __init__(
        self,
        url: str,
        model: str,
        api: str = "openai",
        backend: str = "generic",
        api_key: str | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.root, self.openai_base = split_base(url)
        self.model = model
        self.api = api
        self.backend = backend
        headers = {"User-Agent": "llm-doctor"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self.http = httpx.Client(headers=headers, transport=transport, timeout=60)

    def close(self) -> None:
        self.http.close()

    def _thinking_off(self, payload: dict) -> None:
        if self.api == "ollama":
            payload["think"] = False
        elif self.backend == "ollama":
            payload["reasoning_effort"] = "none"
        elif self.backend == "llama-server":
            payload["chat_template_kwargs"] = {"enable_thinking": False}

    def chat(
        self,
        messages: list[dict],
        *,
        max_tokens: int,
        tools: list[dict] | None = None,
        response_format: dict | None = None,
        stream: bool = False,
        thinking: bool | None = False,
        timeout: float = 60.0,
    ) -> ChatResult:
        if self.api == "ollama":
            return self._ollama(
                messages, max_tokens, tools, response_format, stream, thinking, timeout
            )
        payload: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": stream,
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if tools:
            payload["tools"] = tools
        if response_format:
            payload["response_format"] = response_format
        if thinking is False:
            self._thinking_off(payload)
        url = f"{self.openai_base}/chat/completions"
        t0 = time.monotonic()
        try:
            if stream:
                return self._openai_stream(url, payload, t0, timeout)
            r = self.http.post(url, json=payload, timeout=timeout)
        except httpx.TimeoutException:
            return ChatResult(
                ok=False, error=f"timed out after {timeout:.0f} s", elapsed=time.monotonic() - t0
            )
        except httpx.HTTPError as e:
            return ChatResult(
                ok=False, error=f"{type(e).__name__}: {e}", elapsed=time.monotonic() - t0
            )
        res = ChatResult(
            ok=r.status_code == 200, status=r.status_code, elapsed=time.monotonic() - t0
        )
        if r.status_code != 200:
            res.error = _error_text(r)
            return res
        try:
            data = r.json()
        except ValueError:
            res.ok, res.error = False, "response is not JSON"
            return res
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        res.content = msg.get("content") or ""
        for key in ("reasoning_content", "reasoning"):
            if msg.get(key):
                res.reasoning, res.reasoning_field = msg[key], key
                break
        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function") or {}
            args = fn.get("arguments")
            if not isinstance(args, str):
                args = json.dumps(args)
            res.tool_calls.append(
                ToolCall(name=fn.get("name") or "", arguments=args or "", id=tc.get("id"))
            )
        res.finish_reason = choice.get("finish_reason")
        _usage(res, data)
        return res

    def _openai_stream(self, url: str, payload: dict, t0: float, timeout: float) -> ChatResult:
        res = ChatResult(ok=False)
        calls: dict[int, dict] = {}
        with self.http.stream("POST", url, json=payload, timeout=timeout) as r:
            res.status = r.status_code
            if r.status_code != 200:
                r.read()
                res.error = _error_text(r)
                res.elapsed = time.monotonic() - t0
                return res
            for line in r.iter_lines():
                if time.monotonic() - t0 > timeout:
                    res.error = f"stream exceeded {timeout:.0f} s"
                    res.elapsed = time.monotonic() - t0
                    return res
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    chunk = json.loads(body)
                except ValueError:
                    continue
                now = time.monotonic() - t0
                _usage(res, chunk)
                for choice in chunk.get("choices") or []:
                    delta = choice.get("delta") or {}
                    got = False
                    if delta.get("content"):
                        res.content += delta["content"]
                        res.content_chunks += 1
                        got = True
                    for key in ("reasoning_content", "reasoning"):
                        if delta.get(key):
                            res.reasoning += delta[key]
                            res.reasoning_field = key
                            got = True
                    for tc in delta.get("tool_calls") or []:
                        res.tool_call_chunks += 1
                        got = True
                        slot = calls.setdefault(
                            int(tc.get("index") or 0), {"name": "", "arguments": "", "id": None}
                        )
                        fn = tc.get("function") or {}
                        if tc.get("id"):
                            slot["id"] = tc["id"]
                        if fn.get("name"):
                            slot["name"] += fn["name"]
                        if fn.get("arguments"):
                            a = fn["arguments"]
                            slot["arguments"] += a if isinstance(a, str) else json.dumps(a)
                    if got:
                        if res.first_token_at is None:
                            res.first_token_at = now
                            res.ttft = now
                        res.last_token_at = now
                    if choice.get("finish_reason"):
                        res.finish_reason = choice["finish_reason"]
        res.tool_calls = [ToolCall(**calls[i]) for i in sorted(calls)]
        res.ok = True
        res.elapsed = time.monotonic() - t0
        return res

    def _ollama(
        self, messages, max_tokens, tools, response_format, stream, thinking, timeout
    ) -> ChatResult:
        payload: dict = {
            "model": self.model,
            "messages": [_to_ollama_message(m) for m in messages],
            "stream": stream,
            "options": {"num_predict": max_tokens, "temperature": 0},
        }
        if tools:
            payload["tools"] = tools
        if response_format and response_format.get("type") == "json_schema":
            payload["format"] = response_format["json_schema"]["schema"]
        elif response_format:
            payload["format"] = "json"
        if thinking is False:
            payload["think"] = False
        t0 = time.monotonic()
        res = ChatResult(ok=False)
        try:
            with self.http.stream(
                "POST", f"{self.root}/api/chat", json=payload, timeout=timeout
            ) as r:
                res.status = r.status_code
                if r.status_code != 200:
                    r.read()
                    res.error = _error_text(r)
                    res.elapsed = time.monotonic() - t0
                    return res
                for line in r.iter_lines():
                    if time.monotonic() - t0 > timeout:
                        res.error = f"stream exceeded {timeout:.0f} s"
                        res.elapsed = time.monotonic() - t0
                        return res
                    if not line.strip():
                        continue
                    try:
                        chunk = json.loads(line)
                    except ValueError:
                        continue
                    now = time.monotonic() - t0
                    msg = chunk.get("message") or {}
                    got = False
                    if msg.get("content"):
                        res.content += msg["content"]
                        res.content_chunks += 1
                        got = True
                    if msg.get("thinking"):
                        res.reasoning += msg["thinking"]
                        res.reasoning_field = "thinking"
                        got = True
                    for tc in msg.get("tool_calls") or []:
                        res.tool_call_chunks += 1
                        got = True
                        fn = tc.get("function") or {}
                        args = fn.get("arguments")
                        res.tool_calls.append(
                            ToolCall(
                                name=fn.get("name") or "",
                                arguments=args if isinstance(args, str) else json.dumps(args or {}),
                            )
                        )
                    if got:
                        if res.first_token_at is None:
                            res.first_token_at = res.ttft = now
                        res.last_token_at = now
                    if chunk.get("done"):
                        res.finish_reason = chunk.get("done_reason")
                        res.prompt_tokens = chunk.get("prompt_eval_count")
                        res.completion_tokens = chunk.get("eval_count")
        except httpx.TimeoutException:
            res.error = f"timed out after {timeout:.0f} s"
            res.elapsed = time.monotonic() - t0
            return res
        except httpx.HTTPError as e:
            res.error = f"{type(e).__name__}: {e}"
            res.elapsed = time.monotonic() - t0
            return res
        res.ok = True
        res.elapsed = time.monotonic() - t0
        return res


def _to_ollama_message(m: dict) -> dict:
    out = {"role": m["role"], "content": m.get("content") or ""}
    if m.get("tool_calls"):
        out["tool_calls"] = [
            {
                "function": {
                    "name": tc["function"]["name"],
                    "arguments": json.loads(tc["function"]["arguments"] or "{}"),
                }
            }
            for tc in m["tool_calls"]
        ]
    if m["role"] == "tool" and m.get("name"):
        out["tool_name"] = m["name"]
    return out


def _usage(res: ChatResult, data: dict) -> None:
    usage = data.get("usage") or {}
    if usage.get("prompt_tokens") is not None:
        res.prompt_tokens = usage["prompt_tokens"]
    if usage.get("completion_tokens") is not None:
        res.completion_tokens = usage["completion_tokens"]
    details = usage.get("prompt_tokens_details") or {}
    if details.get("cached_tokens") is not None:
        res.cached_tokens = details["cached_tokens"]
    timings = data.get("timings") or {}  # llama-server
    if timings.get("cache_n") is not None:
        res.cached_tokens = timings["cache_n"]
        if res.prompt_tokens is None and timings.get("prompt_n") is not None:
            res.prompt_tokens = timings["prompt_n"] + timings["cache_n"]


def _error_text(r: httpx.Response) -> str:
    try:
        data = r.json()
        err = data.get("error")
        if isinstance(err, dict):
            err = err.get("message")
        text = str(err or data)
    except ValueError:
        text = r.text
    return f"HTTP {r.status_code}: {text.strip()[:300]}"
