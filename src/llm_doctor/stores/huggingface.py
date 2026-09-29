"""Hugging Face hub cache: models--<org>--<name>/{blobs,refs,snapshots}.

Snapshot files are symlinks into blobs/. For files stored with Git LFS the blob file name
is the sha256 of the content, so weights in the HF cache never need to be hashed.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
from pathlib import Path

from llm_doctor.gguf_meta import GGUFError, read_header
from llm_doctor.memory import arch_from_config, arch_from_gguf
from llm_doctor.stores.base import (
    WEIGHT_SUFFIXES,
    Leftover,
    Model,
    StoreResult,
    WeightFile,
    file_mtime,
)

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
SHARD_RE = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$")


def default_roots() -> list[Path]:
    if os.environ.get("HF_HUB_CACHE"):
        return [Path(os.environ["HF_HUB_CACHE"]).expanduser()]
    if os.environ.get("HF_HOME"):
        return [Path(os.environ["HF_HOME"]).expanduser() / "hub"]
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return [base / "huggingface" / "hub"]


def repo_id_from_dir(name: str) -> str | None:
    if not name.startswith("models--"):
        return None
    return name[len("models--") :].replace("--", "/")


def _snapshot_order(repo_dir: Path) -> list[Path]:
    """Snapshot directories, the one refs/main points to first, then newest first."""
    snaps = repo_dir / "snapshots"
    if not snaps.is_dir():
        return []
    dirs = [d for d in snaps.iterdir() if d.is_dir()]
    main = None
    with contextlib.suppress(OSError):
        main = (repo_dir / "refs" / "main").read_text().strip()
    dirs.sort(key=lambda d: (d.name != main, -d.stat().st_mtime))
    return dirs


def _blob_digest(link: Path, repo_blobs: Path) -> str | None:
    """Blob name (sha256 for LFS files, git sha1 otherwise) from the first symlink hop.

    Snapshot files link to <repo>/blobs/<sha256 or git sha1>. Since huggingface_hub 2.0
    that blob can itself link to a shared, xet-hashed file in <hub>/blobs/xx/<xet hash>.
    The xet hash is also 64 hex characters, so only the first hop names the sha256.
    """
    try:
        first = (
            Path(os.path.normpath(link.parent / os.readlink(link))) if link.is_symlink() else link
        )
    except OSError:
        return None
    if first.parent == repo_blobs:
        return first.name
    return None


def _snapshot_files(snap: Path) -> list[tuple[str, Path, Path | None, str | None]]:
    """(relative name, snapshot path, physical file or None if broken, sha256 if known)."""
    repo_blobs = Path(os.path.normpath(snap.parent.parent / "blobs"))
    out = []
    for dirpath, _dirnames, filenames in os.walk(snap):
        for fn in filenames:
            p = Path(dirpath) / fn
            rel = str(p.relative_to(snap))
            target: Path | None
            try:
                target = p.resolve(strict=True)
            except (OSError, RuntimeError):
                target = None
            out.append((rel, p, target, _blob_digest(p, repo_blobs) if target else None))
    return sorted(out)


def _is_mlx(repo_id: str, cfg: dict | None, rel_names: list[str]) -> bool:
    if repo_id.lower().startswith("mlx-community/") or "mlx" in repo_id.lower().split("/")[-1]:
        return True
    if cfg and isinstance(cfg.get("quantization"), dict) and "group_size" in cfg["quantization"]:
        return True
    return any(n.endswith(".npz") for n in rel_names)


def _load_json(p: Path) -> dict | None:
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else None
    except (OSError, ValueError):
        return None


def read_local_template(folder: Path) -> tuple[str | None, str | None]:
    """Chat template from a transformers-style folder, and which file it came from."""
    jinja = folder / "chat_template.jinja"
    if jinja.exists():
        try:
            return jinja.read_text(), "chat_template.jinja"
        except OSError:
            pass
    tc = _load_json(folder / "tokenizer_config.json")
    if tc:
        t = tc.get("chat_template")
        if isinstance(t, list):  # list of named templates
            t = next((x.get("template") for x in t if x.get("name") == "default"), None)
        if isinstance(t, str):
            return t, "tokenizer_config.json"
    ct = _load_json(folder / "chat_template.json")
    if ct and isinstance(ct.get("chat_template"), str):
        return ct["chat_template"], "chat_template.json"
    return None, None


def scan(root: Path) -> StoreResult:
    res = StoreResult(name="huggingface", root=root)
    if not root.is_dir():
        return res
    for repo_dir in sorted(root.iterdir()):
        repo_id = repo_id_from_dir(repo_dir.name)
        if not repo_id or not repo_dir.is_dir():
            continue
        blobs = repo_dir / "blobs"
        if blobs.is_dir():
            for b in blobs.iterdir():
                if b.name.endswith(".incomplete"):
                    res.leftovers.append(
                        Leftover(
                            store="huggingface",
                            kind="incomplete",
                            label=f"{repo_id}: {b.name[:16]}...incomplete",
                            paths=[b],
                            size=b.stat().st_size,
                            mtime=file_mtime(b),
                        )
                    )
        snaps = _snapshot_order(repo_dir)
        if not snaps:
            continue
        chosen: dict[str, tuple[str, Path, Path | None, str | None]] = {}
        revs: dict[str, str] = {}
        superseded: set[Path] = set()
        for snap in snaps:
            for rel, p, t, dg in _snapshot_files(snap):
                if rel in chosen:
                    if t is not None and t != chosen[rel][2]:
                        superseded.add(t)
                    continue
                chosen[rel] = (rel, p, t, dg)
                revs[rel] = snap.name
        live = {item[2] for item in chosen.values() if item[2] is not None}
        superseded -= live
        files = [chosen[k] for k in sorted(chosen)]
        res.models += _models_for_snapshot(repo_id, repo_dir, snaps[0], files, revs)
        if superseded:
            old = sorted(superseded)
            res.leftovers.append(
                Leftover(
                    store="huggingface",
                    kind="old-revision",
                    label=f"{repo_id}: {len(old)} files superseded by a newer revision",
                    paths=old,
                    size=sum(b.stat().st_size for b in old),
                    mtime=max(file_mtime(b) for b in old),
                )
            )
    return res


def _models_for_snapshot(
    repo_id: str,
    repo_dir: Path,
    snap: Path,
    files: list[tuple[str, Path, Path | None, str | None]],
    revs: dict[str, str],
) -> list[Model]:
    rev = snap.name
    models: list[Model] = []
    broken = [rel for rel, _, t, _ in files if t is None]
    ggufs = [f for f in files if f[0].endswith(".gguf") and f[2] is not None]
    weights = [
        f
        for f in files
        if f[2] is not None and f[0].endswith(WEIGHT_SUFFIXES) and not f[0].endswith(".gguf")
    ]

    def wf(item: tuple, role: str) -> WeightFile:
        _rel, p, t, dg = item
        sha = dg if dg and SHA256_RE.match(dg) else None
        return WeightFile(path=t, display=str(p), size=t.stat().st_size, role=role, digest=sha)

    if ggufs:
        projectors, mains = [], []
        for item in ggufs:
            (projectors if "mmproj" in item[0].lower() else mains).append(item)
        # Group multi-shard files into one model.
        groups: dict[str, list] = {}
        for item in mains:
            m = SHARD_RE.match(item[0])
            groups.setdefault(m.group(1) if m else item[0][: -len(".gguf")], []).append(item)
        for key, items in sorted(groups.items()):
            items.sort()
            first_rel, _first_p, first_t, _ = items[0]
            model = Model(
                store="huggingface",
                name=f"{repo_id}/{Path(key).name}",
                format="gguf",
                location=str(_first_p),
                source_repo=repo_id,
                source_file=first_rel,
                revision=revs.get(first_rel, rev),
                meta={"repo_dir": str(repo_dir)},
            )
            model.files = [wf(it, "model") for it in items]
            model.files += [wf(it, "projector") for it in projectors]
            try:
                h = read_header(first_t)
                model.arch = arch_from_gguf(h)
                model.chat_template = h.chat_template
                model.template_source = "gguf" if h.chat_template else None
                base = h.get("general.base_model.0.repo_url")
                if isinstance(base, str) and "huggingface.co/" in base:
                    model.meta["base_repo"] = base.split("huggingface.co/", 1)[1].strip("/")
            except GGUFError as e:
                model.problems.append(f"unreadable GGUF header: {e}")
            models.append(model)
        return models

    if not weights:
        if broken:
            m = Model(
                store="huggingface",
                name=repo_id,
                format="unknown",
                location=str(snap),
                source_repo=repo_id,
                revision=rev,
            )
            m.problems.append(f"{len(broken)} snapshot files point to missing blobs")
            models.append(m)
        return models

    folder = weights[0][1].parent
    rev = revs.get(weights[0][0], rev)
    cfg = _load_json(folder / "config.json")
    rel_names = [f[0] for f in files]
    fmt = "mlx" if _is_mlx(repo_id, cfg, rel_names) else "safetensors"
    model = Model(
        store="huggingface",
        name=repo_id,
        format=fmt,
        location=str(folder),
        source_repo=repo_id,
        revision=rev,
        meta={"repo_dir": str(repo_dir)},
    )
    model.files = [wf(it, "weights") for it in weights]
    model.meta["files"] = {it[0]: it[3] for it in files if it[2] is not None}
    if cfg:
        model.arch = arch_from_config(cfg)
    model.chat_template, model.template_source = read_local_template(folder)
    if broken:
        model.problems.append(f"{len(broken)} snapshot files point to missing blobs")
    models.append(model)
    return models
