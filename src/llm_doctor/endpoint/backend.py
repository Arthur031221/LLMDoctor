"""Identify the server behind a URL and read what it says about context length."""

from __future__ import annotations

from dataclasses import dataclass, field

import httpx


@dataclass
class BackendInfo:
    kind: str = "generic"  # ollama, llama-server, lmstudio, generic
    version: str | None = None
    models: list[str] = field(default_factory=list)
    model: str | None = None
    advertised_ctx: int | None = None  # what the model supports
    loaded_ctx: int | None = None  # what the server actually allocated
    slots: int | None = None
    arch: dict = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "version": self.version,
            "model": self.model,
            "advertised_context": self.advertised_ctx,
            "loaded_context": self.loaded_ctx,
            "slots": self.slots,
            "notes": self.notes,
        }


def _get_json(http: httpx.Client, url: str) -> dict | list | None:
    try:
        r = http.get(url, timeout=4)
    except httpx.HTTPError:
        return None
    if r.status_code != 200:
        return None
    try:
        return r.json()
    except ValueError:
        return None


def detect(http: httpx.Client, root: str, openai_base: str, model: str | None) -> BackendInfo:
    info = BackendInfo()
    v = _get_json(http, f"{root}/api/version")
    if isinstance(v, dict) and "version" in v:
        info.kind, info.version = "ollama", str(v["version"])
        tags = _get_json(http, f"{root}/api/tags")
        if isinstance(tags, dict):
            info.models = [m.get("name") for m in tags.get("models") or [] if m.get("name")]
        ps = _get_json(http, f"{root}/api/ps")
        loaded = (
            [m.get("name") for m in (ps or {}).get("models") or []] if isinstance(ps, dict) else []
        )
        info.model = model or (loaded[0] if loaded else (info.models[0] if info.models else None))
        if info.model:
            refresh_ollama(http, root, info)
        return info

    props = _get_json(http, f"{root}/props")
    if isinstance(props, dict) and (
        "default_generation_settings" in props or "total_slots" in props
    ):
        info.kind = "llama-server"
        info.version = str(props.get("build_info") or "") or None
        gen = props.get("default_generation_settings") or {}
        info.loaded_ctx = gen.get("n_ctx") or props.get("n_ctx")
        info.slots = props.get("total_slots")
    else:
        lm = _get_json(http, f"{root}/api/v0/models")
        if (
            isinstance(lm, dict)
            and isinstance(lm.get("data"), list)
            and any("max_context_length" in m for m in lm["data"])
        ):
            info.kind = "lmstudio"
            loaded = [m for m in lm["data"] if m.get("state") == "loaded"]
            pick = next((m for m in lm["data"] if m.get("id") == model), None) or (
                loaded[0] if loaded else None
            )
            if pick:
                info.model = model or pick.get("id")
                info.advertised_ctx = pick.get("max_context_length")
                info.loaded_ctx = pick.get("loaded_context_length")

    models = _get_json(http, f"{openai_base}/models")
    entries = models.get("data") if isinstance(models, dict) else None
    if isinstance(entries, list):
        ids = [m.get("id") for m in entries if isinstance(m, dict) and m.get("id")]
        info.models = info.models or ids
        info.model = model or info.model or (ids[0] if ids else None)
        entry = next((m for m in entries if m.get("id") == info.model), None) or {}
        meta = entry.get("meta") or {}
        info.advertised_ctx = (
            info.advertised_ctx
            or meta.get("n_ctx_train")
            or entry.get("context_length")
            or entry.get("max_model_len")
            or entry.get("context_window")
        )
    info.model = info.model or model
    return info


def refresh_ollama(http: httpx.Client, root: str, info: BackendInfo) -> None:
    """Read the model's trained context from /api/show and the loaded context from /api/ps."""
    try:
        r = http.post(f"{root}/api/show", json={"model": info.model}, timeout=5)
        show = r.json() if r.status_code == 200 else {}
    except (httpx.HTTPError, ValueError):
        show = {}
    mi = show.get("model_info") or {}
    arch = mi.get("general.architecture")
    if arch:
        info.advertised_ctx = mi.get(f"{arch}.context_length")
        info.arch = {
            "layers": mi.get(f"{arch}.block_count"),
            "kv_heads": mi.get(f"{arch}.attention.head_count_kv"),
            "heads": mi.get(f"{arch}.attention.head_count"),
            "key_length": mi.get(f"{arch}.attention.key_length"),
            "value_length": mi.get(f"{arch}.attention.value_length"),
            "embedding": mi.get(f"{arch}.embedding_length"),
        }
    params = show.get("parameters") or ""
    for line in params.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "num_ctx" and parts[1].isdigit():
            info.notes.append(f"Modelfile sets num_ctx {parts[1]}")
    ps = _get_json(http, f"{root}/api/ps")
    for m in (ps or {}).get("models") or [] if isinstance(ps, dict) else []:
        if m.get("name") == info.model or m.get("model") == info.model:
            info.loaded_ctx = m.get("context_length") or info.loaded_ctx
            info.arch["resident_bytes"] = m.get("size")
            info.arch["vram_bytes"] = m.get("size_vram")


def kv_bytes_per_token(arch: dict) -> int | None:
    layers, kv = arch.get("layers"), arch.get("kv_heads")
    if isinstance(kv, list):
        kv = max(kv) if kv else None
    k = arch.get("key_length")
    v = arch.get("value_length")
    if not k and arch.get("embedding") and arch.get("heads"):
        heads = arch["heads"] if not isinstance(arch["heads"], list) else max(arch["heads"])
        k = arch["embedding"] // heads
    v = v or k
    if not (layers and kv and k):
        return None
    return int(layers) * int(kv) * (int(k) + int(v)) * 2
