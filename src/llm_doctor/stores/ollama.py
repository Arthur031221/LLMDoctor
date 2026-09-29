"""Ollama model store: manifests/<host>/<namespace>/<model>/<tag> plus blobs/sha256-<hex>.

Each manifest is an OCI-style JSON document. Layers carry a media type:
  application/vnd.ollama.image.model       GGUF weights (or other weights on newer engines)
  application/vnd.ollama.image.projector   vision projector (mmproj GGUF)
  application/vnd.ollama.image.adapter     LoRA adapter
  application/vnd.ollama.image.params      JSON sampling parameters (num_ctx, stop, ...)
  application/vnd.ollama.image.template    Go text/template chat template
  application/vnd.ollama.image.system      system prompt
Unknown media types are kept and reported so newer layouts do not break the scan.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from llm_doctor.gguf_meta import GGUFError, is_gguf, read_header
from llm_doctor.memory import arch_from_gguf
from llm_doctor.stores.base import Leftover, Model, StoreResult, WeightFile, file_mtime

DEFAULT_HOST = "registry.ollama.ai"
MT_PREFIX = "application/vnd.ollama.image."
PARTIAL_RE = re.compile(r"^(sha256-[0-9a-f]{64})-partial(?:-\d+)?$")
BLOB_RE = re.compile(r"^sha256-([0-9a-f]{64})$")


def default_roots() -> list[Path]:
    env = os.environ.get("OLLAMA_MODELS")
    if env:
        return [Path(env).expanduser()]
    roots = [Path.home() / ".ollama" / "models"]
    for extra in (Path("/usr/share/ollama/.ollama/models"), Path("/var/lib/ollama/models")):
        if extra.is_dir():
            roots.append(extra)
    return roots


@dataclass
class OllamaManifest:
    name: str  # display name, e.g. qwen3:1.7b or hf.co/unsloth/Qwen3-0.6B-GGUF:Q4_K_M
    host: str
    namespace: str
    model: str
    tag: str
    path: Path
    digest: str  # sha256 of the manifest bytes, the ID `ollama list` shows
    config_digest: str | None = None
    layers: list[dict] = field(default_factory=list)
    all_digests: set[str] = field(default_factory=set)

    def layers_of(self, kind: str) -> list[dict]:
        return [ly for ly in self.layers if ly.get("mediaType") == MT_PREFIX + kind]


def display_name(host: str, namespace: str, model: str, tag: str) -> str:
    if host == DEFAULT_HOST and namespace == "library":
        base = model
    elif host == DEFAULT_HOST:
        base = f"{namespace}/{model}"
    else:
        base = f"{host}/{namespace}/{model}"
    return f"{base}:{tag}"


def _collect_digests(obj, out: set[str]) -> None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "digest" and isinstance(v, str) and v.startswith("sha256:"):
                out.add(v.split(":", 1)[1])
            else:
                _collect_digests(v, out)
    elif isinstance(obj, list):
        for v in obj:
            _collect_digests(v, out)


def read_manifests(root: Path) -> tuple[list[OllamaManifest], list[str]]:
    import hashlib

    manifests: list[OllamaManifest] = []
    problems: list[str] = []
    mroot = root / "manifests"
    if not mroot.is_dir():
        return manifests, problems
    for p in sorted(mroot.rglob("*")):
        if not p.is_file() or p.name.startswith("."):
            continue
        rel = p.relative_to(mroot).parts
        if len(rel) < 4:
            continue
        host, namespace, model, tag = rel[0], "/".join(rel[1:-2]), rel[-2], rel[-1]
        try:
            raw = p.read_bytes()
            data = json.loads(raw)
        except (OSError, ValueError) as e:
            problems.append(f"unreadable manifest {p}: {e}")
            continue
        digests: set[str] = set()
        _collect_digests(data, digests)
        cfg = data.get("config") or {}
        manifests.append(
            OllamaManifest(
                name=display_name(host, namespace, model, tag),
                host=host,
                namespace=namespace,
                model=model,
                tag=tag,
                path=p,
                digest=hashlib.sha256(raw).hexdigest(),
                config_digest=(cfg.get("digest") or "").split(":", 1)[-1] or None,
                layers=list(data.get("layers") or []),
                all_digests=digests,
            )
        )
    return manifests, problems


def blob_path(root: Path, digest: str) -> Path:
    return root / "blobs" / f"sha256-{digest.split(':', 1)[-1]}"


def _read_small(path: Path, limit: int = 1 << 20) -> str | None:
    try:
        if path.stat().st_size > limit:
            return None
        return path.read_text(errors="replace")
    except OSError:
        return None


def _source_repo(m: OllamaManifest) -> str | None:
    if m.host in ("hf.co", "huggingface.co"):
        return f"{m.namespace}/{m.model}"
    return None


def scan(root: Path, now: float | None = None) -> StoreResult:
    now = now or time.time()
    res = StoreResult(name="ollama", root=root)
    manifests, problems = read_manifests(root)
    res.problems += problems
    referenced: set[str] = set()
    for m in manifests:
        referenced |= m.all_digests
        model = _model_from_manifest(root, m)
        res.models.append(model)

    blobs_dir = root / "blobs"
    partials: dict[str, list[Path]] = {}
    if blobs_dir.is_dir():
        for entry in os.scandir(blobs_dir):
            name = entry.name
            pm = PARTIAL_RE.match(name)
            if pm:
                partials.setdefault(pm.group(1), []).append(Path(entry.path))
                continue
            bm = BLOB_RE.match(name)
            if not bm:
                continue
            if bm.group(1) not in referenced:
                p = Path(entry.path)
                res.leftovers.append(
                    Leftover(
                        store="ollama",
                        kind="orphan",
                        label=name,
                        paths=[p],
                        size=entry.stat(follow_symlinks=False).st_size,
                        mtime=file_mtime(p),
                    )
                )
    for key, paths in sorted(partials.items()):
        size = sum(p.lstat().st_size for p in paths if p.exists())
        res.leftovers.append(
            Leftover(
                store="ollama",
                kind="incomplete",
                label=f"{key} ({len(paths)} files)",
                paths=sorted(paths),
                size=size,
                mtime=max(file_mtime(p) for p in paths),
            )
        )
    return res


def _model_from_manifest(root: Path, m: OllamaManifest) -> Model:
    model = Model(
        store="ollama",
        name=m.name,
        format="unknown",
        location=str(m.path),
        source_repo=_source_repo(m),
        meta={
            "manifest_digest": m.digest,
            "host": m.host,
            "namespace": m.namespace,
            "model": m.model,
            "tag": m.tag,
            "layers": [
                {
                    "mediaType": ly.get("mediaType"),
                    "digest": ly.get("digest"),
                    "size": ly.get("size"),
                }
                for ly in m.layers
            ],
        },
    )
    fmt = None
    if m.config_digest:
        cfg_text = _read_small(blob_path(root, m.config_digest))
        if cfg_text:
            with contextlib.suppress(ValueError, AttributeError):
                cfg = json.loads(cfg_text)
                fmt = cfg.get("model_format")
                # Newer Ollama models format prompts with a built-in Go renderer instead of
                # a template layer, and may need a minimum Ollama version.
                for key in ("renderer", "parser", "requires"):
                    if cfg.get(key):
                        model.meta[key] = cfg[key]
    for ly in m.layers:
        mt = str(ly.get("mediaType") or "")
        digest = str(ly.get("digest") or "").split(":", 1)[-1]
        bp = blob_path(root, digest)
        exists = bp.exists()
        kind = mt[len(MT_PREFIX) :] if mt.startswith(MT_PREFIX) else mt
        if kind in ("model", "projector", "adapter") or (
            int(ly.get("size") or 0) > 64 * 1024 * 1024 and kind not in ("license",)
        ):
            if not exists:
                model.problems.append(f"missing blob for {kind} layer sha256-{digest[:12]}")
                continue
            role = {"model": "model", "projector": "projector", "adapter": "adapter"}.get(
                kind, "weights"
            )
            model.files.append(
                WeightFile(
                    path=bp,
                    display=str(bp),
                    size=bp.stat().st_size,
                    role=role,
                    digest=digest,
                )
            )
            if kind not in ("model", "projector", "adapter"):
                model.meta.setdefault("unknown_layers", []).append(mt)
        elif kind == "params" and exists:
            text = _read_small(bp)
            try:
                model.meta["params"] = json.loads(text) if text else {}
            except ValueError:
                model.problems.append("params layer is not valid JSON")
        elif kind == "template" and exists:
            model.meta["ollama_template"] = _read_small(bp)
            model.meta["template_digest"] = digest
        elif kind == "system" and exists:
            model.meta["system"] = _read_small(bp)
        elif kind in ("messages",) and exists:
            text = _read_small(bp)
            with contextlib.suppress(ValueError):
                model.meta["messages"] = json.loads(text) if text else []
        elif not exists and kind != "license":
            model.problems.append(f"missing blob for {kind} layer sha256-{digest[:12]}")

    main = next((f for f in model.files if f.role == "model"), None)
    if main is None:
        main = next((f for f in model.files if f.role == "weights"), None)
    if main is not None and is_gguf(main.path):
        fmt = "gguf"
        try:
            h = read_header(main.path)
            model.arch = arch_from_gguf(h)
            model.chat_template = h.chat_template
            model.template_source = "gguf" if h.chat_template else None
            base = h.get("general.base_model.0.repo_url")
            if isinstance(base, str) and "huggingface.co/" in base:
                model.meta["base_repo"] = base.split("huggingface.co/", 1)[1].strip("/")
            # Ollama's own engine can keep the vision tower inside the main GGUF.
            # llama.cpp expects it in a separate mmproj file.
            vision = f"{h.arch}.vision."
            has_projector = any(f.role == "projector" for f in model.files)
            if not has_projector and any(k.startswith(vision) for k in h.metadata):
                model.meta["embedded_vision"] = True
        except GGUFError as e:
            model.problems.append(f"unreadable GGUF header: {e}")
    model.format = fmt or ("safetensors" if model.files else "unknown")
    if not model.files and not model.problems:
        model.problems.append("manifest has no weight layers")
    return model
