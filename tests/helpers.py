"""Builders for synthetic model stores. No real weights, every file is a few KB."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import gguf
import numpy as np

TEMPLATE_OLD = "{% for m in messages %}{{ m.content }}{% endfor %}"
TEMPLATE_NEW = "{% for m in messages %}{{ m.content if m.content is not none else '' }}{% endfor %}"


def write_gguf(
    path: Path,
    *,
    arch: str = "qwen3",
    layers: int = 28,
    heads: int = 16,
    kv_heads: int = 8,
    head_dim: int = 128,
    ctx: int = 40960,
    template: str | None = TEMPLATE_OLD,
    vocab: int = 2000,
    tensor_floats: int = 1024,
    mmproj: bool = False,
    base_repo: str | None = None,
    embedded_vision: bool = False,
    seed: int = 0,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(path), arch="clip" if mmproj else arch)
    if mmproj:
        w.add_type("mmproj")
    else:
        w.add_block_count(layers)
        w.add_context_length(ctx)
        w.add_embedding_length(heads * head_dim)
        w.add_head_count(heads)
        w.add_head_count_kv(kv_heads)
        w.add_key_length(head_dim)
        w.add_value_length(head_dim)
        w.add_token_list([f"tok{i}" for i in range(vocab)])
        if template:
            w.add_chat_template(template)
        if embedded_vision:
            w.add_uint32(f"{arch}.vision.block_count", 24)
        if base_repo:
            w.add_base_model_count(1)
            w.add_base_model_repo_url(0, f"https://huggingface.co/{base_repo}")
    rng = np.random.default_rng(seed)
    w.add_tensor("weights", rng.standard_normal(tensor_floats).astype(np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_ollama_blob(root: Path, data: bytes) -> str:
    digest = hashlib.sha256(data).hexdigest()
    p = root / "blobs" / f"sha256-{digest}"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return digest


def add_ollama_model(
    root: Path,
    name: str,
    gguf_bytes: bytes,
    *,
    host: str = "registry.ollama.ai",
    namespace: str = "library",
    tag: str = "latest",
    params: dict | None = None,
    template: str = "{{ .Prompt }}",
    system: str | None = None,
    projector: bytes | None = None,
    config_extra: dict | None = None,
    with_template: bool = True,
) -> dict:
    layers = []
    d = add_ollama_blob(root, gguf_bytes)
    layers.append(
        {
            "mediaType": "application/vnd.ollama.image.model",
            "digest": f"sha256:{d}",
            "size": len(gguf_bytes),
        }
    )
    if projector is not None:
        pd = add_ollama_blob(root, projector)
        layers.append(
            {
                "mediaType": "application/vnd.ollama.image.projector",
                "digest": f"sha256:{pd}",
                "size": len(projector),
            }
        )
    if with_template:
        td = add_ollama_blob(root, template.encode())
        layers.append(
            {
                "mediaType": "application/vnd.ollama.image.template",
                "digest": f"sha256:{td}",
                "size": len(template),
            }
        )
    if params is not None:
        raw = json.dumps(params).encode()
        layers.append(
            {
                "mediaType": "application/vnd.ollama.image.params",
                "digest": f"sha256:{add_ollama_blob(root, raw)}",
                "size": len(raw),
            }
        )
    if system:
        layers.append(
            {
                "mediaType": "application/vnd.ollama.image.system",
                "digest": f"sha256:{add_ollama_blob(root, system.encode())}",
                "size": len(system),
            }
        )
    cfg = json.dumps(
        {"model_format": "gguf", "model_family": "qwen3", **(config_extra or {})}
    ).encode()
    manifest = {
        "schemaVersion": 2,
        "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
        "config": {
            "mediaType": "application/vnd.docker.container.image.v1+json",
            "digest": f"sha256:{add_ollama_blob(root, cfg)}",
            "size": len(cfg),
        },
        "layers": layers,
    }
    mp = root / "manifests" / host / namespace / name / tag
    mp.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(manifest).encode()
    mp.write_bytes(raw)
    return {"manifest": manifest, "digest": hashlib.sha256(raw).hexdigest(), "path": mp}


def add_hf_file(
    hub: Path,
    repo: str,
    filename: str,
    data: bytes,
    revision: str = "a" * 40,
    main: bool = True,
    xet: bool = False,
) -> Path:
    """Put a file in a Hugging Face cache layout. With xet=True, mimic huggingface_hub 2.0:
    the repo blob is a symlink into a shared <hub>/blobs/xx/<xet hash> file."""
    repo_dir = hub / ("models--" + repo.replace("/", "--"))
    blobs = repo_dir / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(data).hexdigest()
    blob = blobs / digest
    if not blob.exists():
        if xet:
            xh = hashlib.sha256(b"xet" + data).hexdigest()
            shared = hub / "blobs" / xh[:2] / xh
            shared.parent.mkdir(parents=True, exist_ok=True)
            shared.write_bytes(data)
            os.symlink(f"../../blobs/{xh[:2]}/{xh}", blob)
        else:
            blob.write_bytes(data)
    snap = repo_dir / "snapshots" / revision / filename
    snap.parent.mkdir(parents=True, exist_ok=True)
    depth = len(Path(filename).parts)
    os.symlink("../" * (depth + 1) + f"blobs/{digest}", snap)
    if main:
        (repo_dir / "refs").mkdir(exist_ok=True)
        (repo_dir / "refs" / "main").write_text(revision)
    return snap


def age(path: Path, seconds: float) -> None:
    t = time.time() - seconds
    os.utime(path, (t, t), follow_symlinks=False)
