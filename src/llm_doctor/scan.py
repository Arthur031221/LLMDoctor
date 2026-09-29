"""Run every store scanner and turn the results into findings."""

from __future__ import annotations

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from llm_doctor.dedupe import DupGroup, find_duplicates
from llm_doctor.findings import Finding, sort_findings
from llm_doctor.hashcache import HashCache
from llm_doctor.memory import FitEstimate, estimate_fit
from llm_doctor.remote import Remote
from llm_doctor.stores import StoreLocation, StoreResult, discover
from llm_doctor.stores import scan as scan_store
from llm_doctor.stores.base import MIN_DEDUPE_BYTES, Model
from llm_doctor.upstream import check_model
from llm_doctor.util import gpu_wired_limit, human_bytes, human_mem, physical_ram, template_hash

DEFAULT_CTX = 32768
ACTIVE_SECONDS = 15 * 60


@dataclass
class ScanOptions:
    paths: list[Path] = field(default_factory=list)
    ctx: int = DEFAULT_CTX
    online: bool = True
    hash_files: bool = True
    min_size: int = MIN_DEDUPE_BYTES
    ram: int | None = None  # override for tests


@dataclass
class ScanReport:
    locations: list[StoreLocation]
    stores: list[StoreResult]
    duplicates: list[DupGroup]
    fits: dict[str, FitEstimate]
    findings: list[Finding]
    ctx: int
    ram: int | None
    budget: int | None
    budget_source: str
    hashed_bytes: int
    online: bool
    network_errors: list[str]
    elapsed: float

    @property
    def models(self) -> list[Model]:
        return [m for s in self.stores for m in s.models]

    def total_bytes(self) -> int:
        return sum(s.disk_bytes() for s in self.stores)

    def reclaimable(self, now: float | None = None) -> int:
        now = now or time.time()
        dup = sum(g.reclaimable for g in self.duplicates)
        left = sum(
            lo.size
            for s in self.stores
            for lo in s.leftovers
            if lo.kind in ("orphan", "incomplete") and lo.age_seconds(now) >= ACTIVE_SECONDS
        )
        return dup + left

    def to_dict(self) -> dict:
        return {
            "context": self.ctx,
            "ram_bytes": self.ram,
            "gpu_budget_bytes": self.budget,
            "gpu_budget_source": self.budget_source,
            "total_bytes": self.total_bytes(),
            "reclaimable_bytes": self.reclaimable(),
            "hashed_bytes": self.hashed_bytes,
            "online": self.online,
            "network_errors": self.network_errors,
            "elapsed_seconds": round(self.elapsed, 2),
            "stores": [
                {
                    "store": s.name,
                    "root": str(s.root),
                    "models": len(s.models),
                    "bytes": s.disk_bytes(),
                    "problems": s.problems,
                }
                for s in self.stores
            ],
            "models": [_model_dict(m, self.fits.get(model_key(m))) for m in self.models],
            "duplicates": [g.to_dict() for g in self.duplicates],
            "leftovers": [
                {
                    "store": lo.store,
                    "kind": lo.kind,
                    "label": lo.label,
                    "bytes": lo.size,
                    "paths": [str(p) for p in lo.paths],
                    "age_seconds": round(lo.age_seconds(time.time())),
                }
                for s in self.stores
                for lo in s.leftovers
            ],
            "findings": [f.to_dict() for f in self.findings],
        }


def model_key(m: Model) -> str:
    return f"{m.store}:{m.name}"


def _model_dict(m: Model, fit: FitEstimate | None) -> dict:
    return {
        "store": m.store,
        "name": m.name,
        "format": m.format,
        "bytes": m.size,
        "location": m.location,
        "source_repo": m.source_repo,
        "revision": m.revision,
        "arch": m.arch.to_dict() if m.arch else None,
        "chat_template_hash": template_hash(m.chat_template),
        "template_source": m.template_source,
        "fit": fit.to_dict() if fit else None,
        "files": [
            {"path": f.display, "role": f.role, "bytes": f.size, "sha256": f.digest}
            for f in m.files
        ],
        "problems": m.problems,
    }


def run_scan(
    opts: ScanOptions,
    remote: Remote | None = None,
    status: Callable[[str], None] | None = None,
    hash_progress: tuple[Callable[[int], None], Callable[[int], None]] | None = None,
) -> ScanReport:
    t0 = time.time()
    say = status or (lambda _msg: None)
    locations = discover(opts.paths)
    stores: list[StoreResult] = []
    for loc in locations:
        say(f"Scanning {loc.name} at {loc.root}")
        stores.append(scan_store(loc))
    models = [m for s in stores for m in s.models]

    findings: list[Finding] = []
    duplicates: list[DupGroup] = []
    cache = HashCache()
    if opts.hash_files:
        say("Looking for duplicate weights")
        on_start, on_progress = hash_progress or (None, None)
        duplicates = find_duplicates(
            models, cache, opts.min_size, progress=on_progress, on_hash_start=on_start
        )
        cache.save()
    # Files hashed now or in an earlier run get their digest, so upstream checks can use it.
    for m in models:
        for wf in m.files:
            if wf.digest is None:
                wf.digest = cache.lookup(wf.path)
    for g in duplicates:
        stores_involved = sorted({c.store for c in g.copies})
        findings.append(
            Finding(
                severity="warn",
                code="duplicate",
                subject=", ".join(sorted({n for c in g.copies for n in c.models})),
                message=(
                    f"{len(g.copies)} copies of the same {human_bytes(g.size)} file in "
                    f"{', '.join(stores_involved)}"
                ),
                fix="llm-doctor fix --yes",
                bytes=g.reclaimable,
                data={"sha256": g.sha256, "paths": [c.file.display for c in g.copies]},
            )
        )

    now = time.time()
    for s in stores:
        for p in s.problems:
            findings.append(Finding("warn", "store-problem", str(s.root), p, store=s.name))
        for lo in s.leftovers:
            findings.append(_leftover_finding(lo, now))
        for m in s.models:
            for p in m.problems:
                findings.append(
                    Finding("error", "broken", m.name, p, store=m.store, fix=_broken_fix(m))
                )
            if (
                m.format == "gguf"
                and m.store != "ollama"
                and not m.chat_template
                and not m.problems
            ):
                findings.append(
                    Finding(
                        "warn",
                        "template-missing",
                        m.name,
                        "GGUF has no chat template, llama.cpp falls back to a generic one",
                        store=m.store,
                        fix="pass --chat-template-file to llama-server or re-download a newer GGUF",
                    )
                )

    ram = opts.ram if opts.ram is not None else physical_ram()
    budget, budget_source = gpu_wired_limit(ram)
    fits: dict[str, FitEstimate] = {}
    for m in models:
        if m.format in ("gguf", "mlx", "safetensors") and m.arch:
            fit = estimate_fit(m.arch, m.size, opts.ctx, budget, ram)
            fits[model_key(m)] = fit
            if fit.status in ("no-fit", "tight"):
                findings.append(_fit_finding(m, fit, budget_source))

    network_errors: list[str] = []
    if opts.online and models:
        say("Checking upstream versions and chat templates")
        own = remote is None
        remote = remote or Remote()
        try:
            upstream: list[Finding] = []
            with ThreadPoolExecutor(max_workers=8) as pool:
                for result in pool.map(lambda m: _safe_check(m, remote), models):
                    upstream += result
            findings += _merge_base_notes(upstream)
        finally:
            network_errors = list(remote.errors)
            if own:
                remote.close()

    return ScanReport(
        locations=locations,
        stores=stores,
        duplicates=duplicates,
        fits=fits,
        findings=sort_findings(findings),
        ctx=opts.ctx,
        ram=ram,
        budget=budget,
        budget_source=budget_source,
        hashed_bytes=cache.hashed_bytes,
        online=opts.online,
        network_errors=network_errors,
        elapsed=time.time() - t0,
    )


def _merge_base_notes(findings: list[Finding]) -> list[Finding]:
    """The same GGUF often sits in several stores. Report each base-template note once."""
    out: list[Finding] = []
    seen: dict[tuple, Finding] = {}
    for f in findings:
        if f.code != "template-vs-base":
            out.append(f)
            continue
        key = (f.data.get("base_repo"), f.data.get("local_hash"))
        if key in seen:
            prev = seen[key]
            if f.subject not in prev.subject.split(", "):
                prev.subject += f", {f.subject}"
            continue
        seen[key] = f
        out.append(f)
    return out


def _safe_check(m: Model, remote: Remote) -> list[Finding]:
    try:
        return check_model(m, remote)
    except Exception as e:  # a single odd repo must not kill the scan
        remote.errors.append(f"{m.name}: {type(e).__name__}: {e}")
        return []


def _leftover_finding(lo, now: float) -> Finding:
    age = lo.age_seconds(now)
    active = age < ACTIVE_SECONDS
    if lo.kind == "orphan":
        return Finding(
            "info" if active else "warn",
            "orphan",
            lo.label,
            "Ollama blob not referenced by any manifest"
            + (" (written in the last 15 minutes, maybe a pull in progress)" if active else ""),
            store=lo.store,
            fix=None if active else "llm-doctor fix --yes",
            bytes=lo.size,
        )
    if lo.kind == "incomplete":
        return Finding(
            "info" if active else "warn",
            "incomplete",
            lo.label,
            "partial download"
            + (" still being written" if active else f", untouched for {_age(age)}"),
            store=lo.store,
            fix=None if active else "llm-doctor fix --yes",
            bytes=lo.size,
        )
    return Finding(
        "info",
        lo.kind,
        lo.label,
        "disk used by files that the current revision no longer references",
        store=lo.store,
        fix="hf cache prune",
        bytes=lo.size,
    )


def _age(seconds: float) -> str:
    if seconds < 3600:
        return f"{int(seconds // 60)} min"
    if seconds < 86400 * 2:
        return f"{int(seconds // 3600)} h"
    return f"{int(seconds // 86400)} days"


def _broken_fix(m: Model) -> str | None:
    if m.store == "ollama":
        return f"ollama pull {m.name}"
    if m.store == "huggingface" and m.source_repo:
        return f"hf download {m.source_repo}"
    return None


def _fit_finding(m: Model, fit: FitEstimate, budget_source: str) -> Finding:
    parts = (
        f"{human_mem(fit.weights)} weights + {human_mem(fit.kv)} KV cache at {fit.context:,} tokens"
    )
    if fit.status == "no-fit":
        return Finding(
            "error",
            "no-fit",
            m.name,
            f"needs {human_mem(fit.total)} ({parts}), more than {human_mem(fit.ram)} of RAM",
            store=m.store,
            fix="use a smaller quant or a shorter context",
            data=fit.to_dict(),
        )
    return Finding(
        "warn",
        "tight-fit",
        m.name,
        f"needs {human_mem(fit.total)} ({parts}), above the {human_mem(fit.budget)} GPU limit ({budget_source})",
        store=m.store,
        fix="raise the limit with `sudo sysctl iogpu.wired_limit_mb=<MB>` or use a shorter context",
        data=fit.to_dict(),
    )
