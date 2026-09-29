"""Compare local models with their upstream source: newer files and changed chat templates."""

from __future__ import annotations

import difflib

from llm_doctor.findings import Finding
from llm_doctor.remote import Remote
from llm_doctor.stores.base import Model
from llm_doctor.stores.ollama import DEFAULT_HOST, MT_PREFIX
from llm_doctor.util import human_bytes, normalize_template, template_hash

TEMPLATE_FILES = ("chat_template.jinja", "tokenizer_config.json", "chat_template.json")


def _diff_stat(old: str, new: str) -> dict:
    a = normalize_template(old).splitlines()
    b = normalize_template(new).splitlines()
    added = removed = 0
    for line in difflib.unified_diff(a, b, lineterm="", n=0):
        if line.startswith("+") and not line.startswith("+++"):
            added += 1
        elif line.startswith("-") and not line.startswith("---"):
            removed += 1
    return {"lines_added": added, "lines_removed": removed}


def _template_finding(model: Model, local: str, remote: str, where: str, fix: str) -> Finding:
    stat = _diff_stat(local, remote)
    return Finding(
        severity="warn",
        code="template-stale",
        store=model.store,
        subject=model.name,
        message=(
            f"chat template differs from {where} "
            f"(local {template_hash(local)}, upstream {template_hash(remote)}, "
            f"+{stat['lines_added']}/-{stat['lines_removed']} lines)"
        ),
        fix=fix,
        data={"local_hash": template_hash(local), "upstream_hash": template_hash(remote), **stat},
    )


def check_model(model: Model, remote: Remote) -> list[Finding]:
    if model.store == "ollama":
        return check_ollama(model, remote)
    if model.source_repo and model.format == "gguf":
        return check_hf_gguf(model, remote)
    if model.source_repo and model.store == "huggingface":
        return check_hf_folder(model, remote)
    return []


def check_ollama(model: Model, remote: Remote) -> list[Finding]:
    meta = model.meta
    got = remote.ollama_manifest(meta["host"], meta["namespace"], meta["model"], meta["tag"])
    if got is None or got[0] == meta.get("manifest_digest"):
        return _hf_base_note(model, remote)
    _digest, manifest = got
    local = {ly.get("mediaType"): ly.get("digest") for ly in meta.get("layers", [])}
    changed, download = [], 0
    for ly in manifest.get("layers") or []:
        mt = ly.get("mediaType")
        if local.get(mt) != ly.get("digest"):
            changed.append(mt[len(MT_PREFIX) :] if mt and mt.startswith(MT_PREFIX) else str(mt))
            download += int(ly.get("size") or 0)
    findings = []
    only_small = changed and all(c not in ("model", "projector", "adapter") for c in changed)
    findings.append(
        Finding(
            severity="warn",
            code="outdated",
            store="ollama",
            subject=model.name,
            message=(
                f"registry has a newer manifest, changed layers: {', '.join(changed) or 'metadata'}"
                f" ({human_bytes(download)} to download)"
            ),
            fix=f"ollama pull {model.name}",
            bytes=download,
            data={
                "changed_layers": changed,
                "download_bytes": download,
                "template_only": only_small,
            },
        )
    )
    if "template" in changed:
        findings.append(
            Finding(
                severity="warn",
                code="template-stale",
                store="ollama",
                subject=model.name,
                message="the registry ships a different chat template for this tag",
                fix=f"ollama pull {model.name}",
            )
        )
    return findings + _hf_base_note(model, remote)


def _hf_base_note(model: Model, remote: Remote) -> list[Finding]:
    """For `ollama pull hf.co/...` models, compare the template with the base model's."""
    if model.meta.get("host") not in ("hf.co", "huggingface.co") or not model.source_repo:
        return []
    if not model.chat_template:
        return []
    return _base_template_info(
        model, remote, _base_repo(model, remote.hf_model_info(model.source_repo))
    )


def check_hf_gguf(model: Model, remote: Remote) -> list[Finding]:
    repo, fname = model.source_repo, model.source_file
    findings: list[Finding] = []
    main = next((f for f in model.files if f.role == "model"), None)
    if not repo or not fname or main is None:
        return findings
    info = remote.hf_model_info(repo)
    if info is None:
        return findings  # unknown or private repo, or offline
    entry = remote.hf_paths_info(repo, "main", [fname]).get(fname)
    guess = " (repo guessed from folder name)" if model.meta.get("repo_guess") else ""
    if entry is None:
        findings.append(
            Finding(
                severity="info",
                code="upstream-missing",
                store=model.store,
                subject=model.name,
                message=f"{fname} is no longer in {repo} on the Hub{guess}",
            )
        )
        return findings
    lfs = entry.get("lfs") or {}
    up_sha, up_size = lfs.get("oid"), lfs.get("size") or entry.get("size")
    local_sha = main.digest
    if local_sha and up_sha and local_sha == up_sha:
        return findings + _base_template_info(model, remote, _base_repo(model, info))
    changed = (local_sha and up_sha and local_sha != up_sha) or (up_size and up_size != main.size)
    fix = _hf_fix(model)
    if changed:
        findings.append(
            Finding(
                severity="warn",
                code="outdated",
                store=model.store,
                subject=model.name,
                message=f"{repo} has a newer {fname}{guess}",
                fix=fix,
                bytes=int(up_size or 0),
                data={
                    "local_sha256": local_sha,
                    "upstream_sha256": up_sha,
                    "upstream_sha": info.get("sha"),
                },
            )
        )
    header = remote.hf_gguf_header(repo, fname)
    remote_tpl = header.chat_template if header else None
    if remote_tpl is None and info.get("gguf"):
        remote_tpl = info["gguf"].get("chat_template")
    if model.chat_template and remote_tpl:
        if template_hash(model.chat_template) != template_hash(remote_tpl):
            findings.append(
                _template_finding(model, model.chat_template, remote_tpl, f"{repo} on the Hub", fix)
            )
    elif remote_tpl and not model.chat_template:
        findings.append(
            Finding(
                severity="warn",
                code="template-stale",
                store=model.store,
                subject=model.name,
                message=f"local file has no chat template, the current {fname} on the Hub does",
                fix=fix,
            )
        )
    return findings + _base_template_info(model, remote, _base_repo(model, info))


def check_hf_folder(model: Model, remote: Remote) -> list[Finding]:
    repo = model.source_repo
    info = remote.hf_model_info(repo) if repo else None
    if not info or not info.get("sha") or info["sha"] == model.revision:
        return []
    local_files: dict[str, str | None] = model.meta.get("files", {})
    names = [n for n in local_files if n.endswith((".safetensors", ".npz")) or n in TEMPLATE_FILES]
    upstream = remote.hf_paths_info(repo, "main", names)
    changed_weights, changed_tpl = [], []
    for n in names:
        e = upstream.get(n)
        if e is None:
            continue
        up_id = (e.get("lfs") or {}).get("oid") or e.get("oid")
        if up_id and local_files.get(n) and up_id != local_files[n]:
            (changed_tpl if n in TEMPLATE_FILES else changed_weights).append(n)
    findings: list[Finding] = []
    fix = _hf_fix(model)
    if changed_weights:
        findings.append(
            Finding(
                severity="warn",
                code="outdated",
                store=model.store,
                subject=model.name,
                message=f"newer revision {info['sha'][:10]} changes {len(changed_weights)} weight files",
                fix=fix,
                data={"local_revision": model.revision, "upstream_revision": info["sha"]},
            )
        )
    if changed_tpl:
        remote_tpl, where = remote.hf_template(repo)
        local_tpl = model.chat_template
        if local_tpl and remote_tpl and template_hash(local_tpl) != template_hash(remote_tpl):
            findings.append(
                _template_finding(model, local_tpl, remote_tpl, f"{repo} ({where})", fix)
            )
    if not findings:
        findings.append(
            Finding(
                severity="info",
                code="newer-revision",
                store=model.store,
                subject=model.name,
                message=f"repo moved to {info['sha'][:10]}, your weights and template are unchanged",
            )
        )
    return findings


def _base_repo(model: Model, info: dict | None) -> str | None:
    if model.meta.get("base_repo"):
        return model.meta["base_repo"]
    card = (info or {}).get("cardData") or {}
    base = card.get("base_model")
    if isinstance(base, list):
        base = base[0] if len(base) == 1 else None
    return base if isinstance(base, str) and "/" in base else None


def _base_template_info(model: Model, remote: Remote, base: str | None) -> list[Finding]:
    if not base or base == model.source_repo or not model.chat_template:
        return []
    tpl, where = remote.hf_template(base)
    if not tpl or template_hash(tpl) == template_hash(model.chat_template):
        return []
    stat = _diff_stat(model.chat_template, tpl)
    return [
        Finding(
            severity="info",
            code="template-vs-base",
            store=model.store,
            subject=model.name,
            message=(
                f"chat template differs from base model {base} ({where}), "
                f"+{stat['lines_added']}/-{stat['lines_removed']} lines. Quantizers often patch "
                "templates on purpose, so this is only a hint"
            ),
            data={"base_repo": base, "local_hash": template_hash(model.chat_template), **stat},
        )
    ]


def _hf_fix(model: Model) -> str:
    if model.store == "huggingface" and model.source_file:
        return f"hf download {model.source_repo} {model.source_file}"
    if model.store == "huggingface":
        return f"hf download {model.source_repo}"
    if model.store == "lmstudio":
        return f"re-download {model.source_repo} in LM Studio"
    return f"re-download from {model.source_repo}"


def is_default_registry(model: Model) -> bool:
    return model.meta.get("host") == DEFAULT_HOST
