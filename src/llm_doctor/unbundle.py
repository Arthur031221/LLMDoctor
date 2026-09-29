"""Expose Ollama blobs as named GGUF files and write configs for other runtimes.

Nothing is copied. Each GGUF gets a symlink (or hardlink) with a readable name, vision
projectors are paired as mmproj files, and Modelfile parameters are translated into a
llama-server preset INI (--models-preset), a llama-swap config, or an LM Studio folder.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from llm_doctor.stores.base import Model
from llm_doctor.util import slugify

TARGETS = ("llama-server", "llama-swap", "lmstudio", "mlx")

# Ollama Modelfile parameter -> llama-server flag (long form without dashes).
PARAM_MAP = {
    "num_ctx": "ctx-size",
    "temperature": "temp",
    "top_k": "top-k",
    "top_p": "top-p",
    "min_p": "min-p",
    "typical_p": "typical",
    "repeat_penalty": "repeat-penalty",
    "repeat_last_n": "repeat-last-n",
    "presence_penalty": "presence-penalty",
    "frequency_penalty": "frequency-penalty",
    "seed": "seed",
    "num_predict": "n-predict",
    "mirostat": "mirostat",
    "mirostat_eta": "mirostat-lr",
    "mirostat_tau": "mirostat-ent",
    "num_gpu": "n-gpu-layers",
    "num_thread": "threads",
    "num_batch": "batch-size",
    "num_keep": "keep",
}


@dataclass
class Export:
    name: str
    slug: str
    blob: Path
    out_model: Path
    projector_blob: Path | None = None
    out_projector: Path | None = None
    adapter_blobs: list[Path] = field(default_factory=list)
    out_adapters: list[Path] = field(default_factory=list)
    flags: dict = field(default_factory=dict)
    stop: list[str] = field(default_factory=list)
    system: str | None = None
    ctx: int | None = None
    ctx_from_modelfile: bool = False
    base_repo: str | None = None
    notes: list[str] = field(default_factory=list)
    skipped: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "file": str(self.out_model),
            "source_blob": str(self.blob),
            "mmproj": str(self.out_projector) if self.out_projector else None,
            "lora": [str(p) for p in self.out_adapters],
            "flags": self.flags,
            "stop": self.stop,
            "system_prompt": self.system,
            "context": self.ctx,
            "notes": self.notes,
            "skipped": self.skipped,
        }


def default_out(target: str) -> Path:
    if target == "lmstudio":
        from llm_doctor.stores.folders import lmstudio_roots

        roots = lmstudio_roots()
        root = next(
            (r for r in roots if r.is_dir()),
            roots[0] if roots else Path.home() / ".lmstudio" / "models",
        )
        return root / "ollama"
    return Path.cwd() / "ollama-unbundled"


def plan_exports(
    models: list[Model], target: str, out: Path, ctx_default: int, only: list[str] | None = None
) -> list[Export]:
    exports: list[Export] = []
    used: set[str] = set()
    for m in sorted(models, key=lambda m: m.name):
        if m.store != "ollama":
            continue
        if only and m.name not in only and m.name.split(":")[0] not in only:
            continue
        main = next((f for f in m.files if f.role == "model"), None)
        base_slug = slugify(m.name.replace("hf.co/", "").replace("/", "-").replace(":", "-"))
        slug = base_slug
        if slug in used:
            slug = f"{base_slug}-{(main.digest or '')[:8]}"
        used.add(slug)
        ex = Export(name=m.name, slug=slug, blob=main.path if main else Path(), out_model=Path())
        if main is None or m.format != "gguf":
            ex.skipped = (
                f"no GGUF weights layer (format {m.format}), this layout is not supported yet"
            )
            exports.append(ex)
            continue
        proj = next((f for f in m.files if f.role == "projector"), None)
        adapters = [f for f in m.files if f.role == "adapter"]
        folder = out / slug if (proj or target == "lmstudio") else out
        ex.out_model = folder / f"{slug}.gguf"
        if proj:
            ex.projector_blob = proj.path
            ex.out_projector = folder / f"mmproj-{slug}.gguf"
        for i, a in enumerate(adapters):
            ex.adapter_blobs.append(a.path)
            ex.out_adapters.append(folder / f"{slug}.lora{i or ''}.gguf")
        params = dict(m.meta.get("params") or {})
        stop = params.pop("stop", [])
        ex.stop = [stop] if isinstance(stop, str) else list(stop or [])
        for k, v in params.items():
            if k in PARAM_MAP:
                ex.flags[PARAM_MAP[k]] = v
            else:
                ex.notes.append(f"parameter {k}={v} has no llama-server equivalent, dropped")
        if "ctx-size" in ex.flags:
            ex.ctx = int(ex.flags["ctx-size"])
            ex.ctx_from_modelfile = True
        else:
            train = m.arch.ctx_train if m.arch else None
            ex.ctx = min(ctx_default, train) if train else ctx_default
            ex.flags = {"ctx-size": ex.ctx, **ex.flags}
            ex.notes.append(
                f"Modelfile sets no num_ctx (Ollama used its server default), wrote ctx-size {ex.ctx}"
            )
        ex.system = m.meta.get("system")
        if ex.system:
            ex.notes.append("system prompt kept as a comment: llama-server has no flag for it")
        if not m.chat_template:
            ex.notes.append(
                "GGUF has no Jinja chat template, pass --chat-template-file (Ollama used its own Go template)"
            )
        ex.base_repo = m.meta.get("base_repo") or (
            m.source_repo if m.meta.get("host") == "hf.co" else None
        )
        exports.append(ex)
    return exports


def _link(src: Path, dst: Path, mode: str) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return "exists"
        if dst.is_symlink():
            dst.unlink()
        elif dst.stat().st_ino == src.stat().st_ino:
            return "exists"
        else:
            raise FileExistsError(f"{dst} exists and is not a link, not touching it")
    if mode == "hardlink":
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            pass
    os.symlink(src, dst)
    return "symlink"


def _ini_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def render_presets(exports: list[Export]) -> str:
    lines = [
        "; llama-server model presets written by llm-doctor unbundle",
        f"; {time.strftime('%Y-%m-%d')}. Start with: llama-server --models-preset <this file>",
        "version = 1",
        "",
        "[*]",
        "jinja = true",
        "",
    ]
    for ex in exports:
        if ex.skipped:
            continue
        lines.append(f"[{ex.name}]")
        lines.append(f"model = {ex.out_model}")
        if ex.out_projector:
            lines.append(f"mmproj = {ex.out_projector}")
        for a in ex.out_adapters:
            lines.append(f"lora = {a}")
        for k, v in ex.flags.items():
            lines.append(f"{k} = {_ini_value(v)}")
        if ex.stop:
            lines.append(
                "; Ollama stop strings, send them as `stop` in requests: " + json.dumps(ex.stop)
            )
        if ex.system:
            first = ex.system.strip().splitlines()[0][:200] if ex.system.strip() else ""
            lines.append("; Ollama system prompt (send it as the first message): " + first)
        lines.append("")
    return "\n".join(lines)


def _cmd_args(ex: Export, binary: str) -> list[str]:
    args = [binary, "--port", "${PORT}", "--model", str(ex.out_model)]
    if ex.out_projector:
        args += ["--mmproj", str(ex.out_projector)]
    for a in ex.out_adapters:
        args += ["--lora", str(a)]
    args.append("--jinja")
    for k, v in ex.flags.items():
        args += [f"--{k}", _ini_value(v)]
    return args


def render_llama_swap(exports: list[Export], binary: str) -> str:
    lines = [
        "# llama-swap config written by llm-doctor unbundle",
        "# Start with: llama-swap --config <this file>",
        "models:",
    ]
    for ex in exports:
        if ex.skipped:
            continue
        lines.append(f"  {json.dumps(ex.name)}:")
        lines.append("    cmd: |")
        args = _cmd_args(ex, binary)
        parts = [args[0] + " " + " ".join(args[1:3])]
        rest = args[3:]
        i = 0
        while i < len(rest):
            if rest[i].startswith("--") and i + 1 < len(rest) and not rest[i + 1].startswith("--"):
                parts.append(f"{rest[i]} {_quote(rest[i + 1])}")
                i += 2
            else:
                parts.append(rest[i])
                i += 1
        for p in parts:
            lines.append(f"      {p}")
        if ex.stop:
            lines.append("    filters:")
            lines.append("      setParams:")
            lines.append(f'        "stop?": {json.dumps(ex.stop)}')
        if ex.system:
            lines.append("    # Ollama system prompt, send it as the first message:")
            for sl in ex.system.strip().splitlines()[:5]:
                lines.append(f"    #   {sl[:200]}")
    return "\n".join(lines) + "\n"


def _quote(v: str) -> str:
    return v if v == "${PORT}" else shlex.quote(v)


def render_mlx_script(exports: list[Export]) -> str:
    lines = [
        "#!/bin/sh",
        "# mlx-lm cannot load Ollama's GGUF quants directly. These commands rebuild MLX",
        "# weights from the original Hugging Face repos. Written by llm-doctor unbundle.",
        "set -e",
    ]
    for ex in exports:
        if ex.skipped:
            continue
        if ex.base_repo:
            lines.append(f"# {ex.name}")
            lines.append(
                f"mlx_lm.convert --hf-path {shlex.quote(ex.base_repo)} -q "
                f"--mlx-path {shlex.quote(ex.slug + '-mlx')}"
            )
        else:
            lines.append(
                f"# {ex.name}: source repo unknown, find it on huggingface.co and convert it"
            )
    return "\n".join(lines) + "\n"


def write_exports(
    exports: list[Export],
    target: str,
    out: Path,
    link_mode: str = "symlink",
    binary: str | None = None,
) -> dict:
    written: dict = {"links": [], "configs": [], "errors": []}
    out.mkdir(parents=True, exist_ok=True)
    if target == "mlx":
        script = out / "convert-to-mlx.sh"
        script.write_text(render_mlx_script(exports))
        script.chmod(0o755)
        written["configs"].append(str(script))
        return written
    for ex in exports:
        if ex.skipped:
            continue
        pairs = [(ex.blob, ex.out_model)]
        if ex.projector_blob and ex.out_projector:
            pairs.append((ex.projector_blob, ex.out_projector))
        pairs += list(zip(ex.adapter_blobs, ex.out_adapters, strict=False))
        for src, dst in pairs:
            try:
                how = _link(src, dst, link_mode)
                written["links"].append({"path": str(dst), "target": str(src), "mode": how})
            except OSError as e:
                written["errors"].append(f"{dst}: {e}")
                ex.skipped = str(e)
    if target == "llama-server":
        p = out / "presets.ini"
        p.write_text(render_presets(exports))
        written["configs"].append(str(p))
    elif target == "llama-swap":
        binary = binary or shutil.which("llama-server") or "llama-server"
        p = out / "llama-swap.yaml"
        p.write_text(render_llama_swap(exports, binary))
        written["configs"].append(str(p))
    return written
