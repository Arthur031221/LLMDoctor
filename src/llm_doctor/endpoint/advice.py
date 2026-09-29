"""Turn failed checks into concrete fixes for the backend that served them."""

from __future__ import annotations

from llm_doctor.endpoint.backend import BackendInfo, kv_bytes_per_token
from llm_doctor.endpoint.probes import FAIL, WARN, Check
from llm_doctor.util import human_mem

TARGET_CTX = 32768


def fixes(checks: list[Check], info: BackendInfo, api: str) -> list[str]:
    bad = {c.id: c for c in checks if c.status in (FAIL, WARN)}
    out: list[str] = []
    kind = info.kind
    model = info.model or "<model>"

    ctx = bad.get("context")
    truncated = ctx is not None and any(
        "truncation" in str(r.get("result")) or r.get("result") == "rejected"
        for r in ctx.data.get("levels", [])
    )
    if ctx is not None and truncated:
        want = TARGET_CTX
        kv = kv_bytes_per_token(info.arch) if info.arch else None
        mem = (
            f" This model needs about {human_mem(kv * want)} of KV cache at {want:,} tokens."
            if kv
            else ""
        )
        if kind == "ollama":
            loaded = (
                f" Ollama loaded {model} with a {info.loaded_ctx:,}-token window."
                if info.loaded_ctx
                else ""
            )
            out.append(
                f"Raise Ollama's context window.{loaded}{mem} Server-wide: "
                f"`OLLAMA_CONTEXT_LENGTH={want} ollama serve`, or for the menu bar app "
                f"`launchctl setenv OLLAMA_CONTEXT_LENGTH {want}` and restart Ollama."
            )
            out.append(
                f"Or per model: `printf 'FROM {model}\\nPARAMETER num_ctx {want}\\n' > Modelfile && "
                f"ollama create {model.split(':')[0]}-{want // 1024}k -f Modelfile`, then point the agent at the new name."
            )
            if api == "openai":
                out.append(
                    "The OpenAI-compatible /v1 endpoint has no num_ctx field, so a client cannot raise "
                    "the window per request. Ollama's native /api/chat accepts options.num_ctx."
                )
        elif kind == "llama-server":
            slots = f" The server runs {info.slots} slots." if info.slots and info.slots > 1 else ""
            out.append(
                f"Start llama-server with `-c {want}` (or `ctx-size = {want}` in the preset INI).{mem}{slots} "
                "Without a unified KV cache each slot gets ctx-size divided by -np, so use `-np 1` "
                "or `--kv-unified` for a single agent."
            )
        elif kind == "lmstudio":
            out.append(
                f"Reload the model in LM Studio with Context Length {want} in its load settings, or run "
                f"`lms load {model} --context-length {want}`.{mem}"
            )
        else:
            out.append(
                f"Raise the server's context window to at least {want} tokens "
                f"(vLLM `--max-model-len`, llama-server `-c`, Ollama `num_ctx`).{mem}"
            )

    tool_bad = [
        bad[k]
        for k in ("tool-call", "tool-stream", "tool-result")
        if k in bad and bad[k].status == FAIL
    ]
    if tool_bad:
        as_text = any("as text" in c.detail for c in tool_bad)
        if kind == "ollama":
            out.append(
                f"Check `ollama show {model}` lists tools under Capabilities and update Ollama. "
                + (
                    "The model printed its tool call as text, which means the template or parser does not match the model."
                    if as_text
                    else ""
                )
            )
        elif kind == "llama-server":
            out.append(
                "Run llama-server with `--jinja` and a tool-aware template (`--chat-template-file`). "
                + (
                    "Tool calls printed as text mean llama.cpp does not parse this model's call format, update llama.cpp."
                    if as_text
                    else ""
                )
            )
        elif kind == "lmstudio":
            out.append(
                "Use a model LM Studio marks as tool-capable and update LM Studio's runtime."
            )
        else:
            out.append(
                "Enable tool parsing on the server (vLLM `--enable-auto-tool-choice --tool-call-parser <name>`)."
            )
    if "tool-parallel" in bad and bad["tool-parallel"].status == WARN:
        out.append(
            "The model makes one tool call per turn. Agents still work, with more round trips."
        )

    if "think-tags" in bad and bad["think-tags"].status == FAIL:
        if kind == "llama-server":
            out.append("Add `--reasoning-format deepseek` so reasoning moves to reasoning_content.")
        elif kind == "ollama":
            out.append(
                "Update Ollama. Current versions return reasoning in a separate field on /v1 and /api/chat."
            )
        elif kind == "lmstudio":
            out.append(
                "Turn on LM Studio's developer setting that separates reasoning_content from content."
            )
        else:
            out.append(
                "Configure the server's reasoning parser so <think> blocks leave the content field."
            )

    if "json-schema" in bad:
        if kind == "ollama":
            out.append(
                "Structured output: Ollama's /api/chat takes the schema in `format`. Update Ollama if /v1 ignores response_format."
            )
        elif kind == "llama-server":
            out.append(
                "Update llama.cpp. Current builds turn response_format json_schema into a grammar."
            )
        else:
            out.append("Enable structured output (guided decoding) on the server.")

    if "max-tokens" in bad and bad["max-tokens"].status == FAIL:
        out.append(
            "The server ignores max_tokens. Cap generation server-side (llama-server `-n`, Ollama `num_predict`)."
        )

    if "prefix-cache" in bad:
        if kind == "llama-server":
            out.append(
                "Enable prompt reuse: `--cache-reuse 256`, keep `--cache-ram` above 0, and use few slots."
            )
        elif kind == "ollama":
            out.append(
                "Keep the model loaded (`OLLAMA_KEEP_ALIVE=-1`) and avoid swapping models between agent turns."
            )
        else:
            out.append(
                "Enable prefix caching on the server so repeated agent prompts skip prefill."
            )

    if "ram" in bad:
        out.append(
            "Free memory or pick a smaller quant. Swapping model weights makes every token slower."
        )
    return out
