"""Plain folder stores: LM Studio, MLX folders and any --path the user passes.

LM Studio keeps models at <root>/<publisher>/<repo>/<file>.gguf, mirroring Hugging Face
repo names, with vision projectors (mmproj-*.gguf) in the same folder. MLX and other
transformers-style models are folders with config.json next to *.safetensors.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from llm_doctor.gguf_meta import GGUFError, read_header
from llm_doctor.memory import arch_from_config, arch_from_gguf
from llm_doctor.stores.base import (
    PARTIAL_SUFFIXES,
    Leftover,
    Model,
    StoreResult,
    WeightFile,
    file_mtime,
)
from llm_doctor.stores.huggingface import read_local_template

SHARD_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$")
MAX_DEPTH = 6


def lmstudio_roots() -> list[Path]:
    roots: list[Path] = []
    pointer = Path.home() / ".lmstudio-home-pointer"
    try:
        target = Path(pointer.read_text().strip()).expanduser()
        if target.is_dir():
            roots.append(target / "models")
    except OSError:
        pass
    for p in (
        Path.home() / ".lmstudio" / "models",
        Path.home() / ".cache" / "lm-studio" / "models",
    ):
        if p not in roots:
            roots.append(p)
    return roots


def mlx_roots() -> list[Path]:
    return [Path.home() / "mlx_models", Path.home() / "mlx-models"]


def extra_roots_from_env() -> list[Path]:
    raw = os.environ.get("LLM_DOCTOR_PATHS", "")
    return [Path(p).expanduser() for p in raw.split(os.pathsep) if p.strip()]


def _load_json(p: Path) -> dict | None:
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def _wf(p: Path, root: Path, role: str) -> WeightFile:
    return WeightFile(path=p, display=str(p), size=p.stat().st_size, role=role)


def scan(root: Path, store: str) -> StoreResult:
    res = StoreResult(name=store, root=root)
    if not root.is_dir():
        return res
    root_depth = len(root.parts)
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        d = Path(dirpath)
        depth = len(d.parts) - root_depth
        dirnames[:] = sorted(n for n in dirnames if not n.startswith("."))
        if depth >= MAX_DEPTH:
            dirnames[:] = []
        files = sorted(filenames)
        for fn in files:
            if fn.endswith(PARTIAL_SUFFIXES):
                p = d / fn
                res.leftovers.append(
                    Leftover(
                        store=store,
                        kind="incomplete",
                        label=str(p.relative_to(root)),
                        paths=[p],
                        size=p.lstat().st_size,
                        mtime=file_mtime(p),
                    )
                )
        ggufs = [fn for fn in files if fn.endswith(".gguf")]
        if ggufs:
            res.models += _gguf_models(root, d, ggufs, store)
        if "config.json" in files and any(fn.endswith(".safetensors") for fn in files):
            res.models.append(_folder_model(root, d, files, store))
    return res


def _repo_guess(root: Path, d: Path, store: str) -> str | None:
    if store != "lmstudio":
        return None
    parts = d.relative_to(root).parts
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return None


def _gguf_models(root: Path, d: Path, ggufs: list[str], store: str) -> list[Model]:
    projectors = [fn for fn in ggufs if "mmproj" in fn.lower()]
    mains = [fn for fn in ggufs if fn not in projectors]
    groups: dict[str, list[str]] = {}
    for fn in mains:
        m = SHARD_RE.match(fn)
        groups.setdefault(m.group(1) if m else fn[: -len(".gguf")], []).append(fn)
    out = []
    for key, names in sorted(groups.items()):
        names.sort()
        first = d / names[0]
        rel_dir = d.relative_to(root)
        name = str(rel_dir / key) if str(rel_dir) != "." else key
        model = Model(
            store=store,
            name=name,
            format="gguf",
            location=str(first),
            source_repo=_repo_guess(root, d, store),
            source_file=names[0],
        )
        try:
            model.files = [_wf(d / n, root, "model") for n in names]
            model.files += [_wf(d / n, root, "projector") for n in projectors]
        except OSError as e:
            model.problems.append(f"cannot stat file: {e}")
            out.append(model)
            continue
        if model.source_repo:
            model.meta["repo_guess"] = True
        try:
            h = read_header(first)
            if h.is_projector:
                continue
            model.arch = arch_from_gguf(h)
            model.chat_template = h.chat_template
            model.template_source = "gguf" if h.chat_template else None
            base = h.get("general.base_model.0.repo_url")
            if isinstance(base, str) and "huggingface.co/" in base:
                model.meta["base_repo"] = base.split("huggingface.co/", 1)[1].strip("/")
        except GGUFError as e:
            model.problems.append(f"unreadable GGUF header: {e}")
        except OSError as e:
            model.problems.append(f"cannot read file: {e}")
        out.append(model)
    return out


def _folder_model(root: Path, d: Path, files: list[str], store: str) -> Model:
    cfg = _load_json(d / "config.json") or {}
    rel = d.relative_to(root)
    lowered = str(rel).lower()
    is_mlx = isinstance(cfg.get("quantization"), dict) or "mlx" in lowered or store == "mlx"
    model = Model(
        store=store,
        name=str(rel) if str(rel) != "." else d.name,
        format="mlx" if is_mlx else "safetensors",
        location=str(d),
        source_repo=_repo_guess(root, d, store),
    )
    for fn in files:
        if fn.endswith(".safetensors"):
            try:
                model.files.append(_wf(d / fn, root, "weights"))
            except OSError as e:
                model.problems.append(f"cannot stat {fn}: {e}")
    model.arch = arch_from_config(cfg) if cfg else None
    model.chat_template, model.template_source = read_local_template(d)
    return model
