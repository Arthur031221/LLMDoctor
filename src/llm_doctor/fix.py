"""Plan and apply fixes: dedupe with links, prune orphan blobs and stale partial downloads."""

from __future__ import annotations

import contextlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from llm_doctor.hashcache import HashCache
from llm_doctor.scan import ScanReport
from llm_doctor.util import doctor_home

KINDS = ("dedupe", "orphans", "incomplete")


@dataclass
class Action:
    kind: str  # link, delete
    category: str  # dedupe, orphans, incomplete
    store: str
    path: Path
    bytes: int
    reason: str
    target: Path | None = None
    extra_paths: list[Path] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "action": self.kind,
            "category": self.category,
            "store": self.store,
            "path": str(self.path),
            "target": str(self.target) if self.target else None,
            "extra_paths": [str(p) for p in self.extra_paths],
            "bytes": self.bytes,
            "reason": self.reason,
        }


@dataclass
class Skip:
    path: str
    reason: str

    def to_dict(self) -> dict:
        return {"path": self.path, "reason": self.reason}


@dataclass
class FixPlan:
    actions: list[Action]
    skipped: list[Skip]
    link_mode: str

    @property
    def reclaim_bytes(self) -> int:
        return sum(a.bytes for a in self.actions)

    def to_dict(self) -> dict:
        return {
            "link_mode": self.link_mode,
            "reclaim_bytes": self.reclaim_bytes,
            "actions": [a.to_dict() for a in self.actions],
            "skipped": [s.to_dict() for s in self.skipped],
        }


def plan_fix(
    report: ScanReport,
    link_mode: str = "symlink",
    min_age: float = 3600,
    only: set[str] | None = None,
    now: float | None = None,
) -> FixPlan:
    now = now or time.time()
    only = only or set(KINDS)
    actions: list[Action] = []
    skipped: list[Skip] = []
    cache = HashCache()

    if "dedupe" in only:
        for g in report.duplicates:
            keep = g.keeper()
            target = Path(keep.file.display) if keep.store == "huggingface" else keep.file.path
            for c in g.copies:
                if c is keep:
                    continue
                if c.store == "huggingface":
                    skipped.append(
                        Skip(
                            c.file.display,
                            "Hugging Face cache files are managed by huggingface_hub",
                        )
                    )
                    continue
                if c.file.is_link():
                    skipped.append(Skip(c.file.display, "already a link"))
                    continue
                if c.file.digest != g.sha256 and cache.lookup(c.file.path) != g.sha256:
                    skipped.append(Skip(c.file.display, "file changed since it was hashed"))
                    continue
                actions.append(
                    Action(
                        kind="link",
                        category="dedupe",
                        store=c.store,
                        path=c.file.path,
                        target=target,
                        bytes=g.size,
                        reason=f"same sha256 as {keep.store} copy ({g.sha256[:12]})",
                    )
                )

    for s in report.stores:
        for lo in s.leftovers:
            cat = {"orphan": "orphans", "incomplete": "incomplete"}.get(lo.kind)
            if cat is None or cat not in only:
                continue
            if lo.age_seconds(now) < min_age:
                skipped.append(
                    Skip(
                        lo.label,
                        f"modified {int(lo.age_seconds(now) // 60)} min ago, younger than --min-age",
                    )
                )
                continue
            actions.append(
                Action(
                    kind="delete",
                    category=cat,
                    store=lo.store,
                    path=lo.paths[0],
                    extra_paths=lo.paths[1:],
                    bytes=lo.size,
                    reason="Ollama blob not referenced by any manifest"
                    if lo.kind == "orphan"
                    else "abandoned partial download",
                )
            )
    return FixPlan(actions=actions, skipped=skipped, link_mode=link_mode)


def _replace_with_link(path: Path, target: Path, mode: str) -> str:
    tmp = path.with_name(f".{path.name}.llm-doctor-tmp")
    with contextlib.suppress(FileNotFoundError):
        tmp.unlink()
    used = mode
    if mode == "hardlink":
        try:
            os.link(target.resolve(), tmp)
        except OSError:
            used = "symlink"  # different filesystem
    if used == "symlink":
        os.symlink(target, tmp)
    os.replace(tmp, path)
    return used


def apply_fix(plan: FixPlan) -> list[dict]:
    results = []
    log_path = doctor_home() / "fix-log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "a") as log:
        for a in plan.actions:
            entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S"), **a.to_dict()}
            try:
                if a.kind == "link":
                    assert a.target is not None
                    if not a.target.exists():
                        raise FileNotFoundError(f"keeper {a.target} is gone")
                    if a.target.stat().st_size != a.path.stat().st_size:
                        raise ValueError("sizes differ, refusing to link")
                    entry["link_mode"] = _replace_with_link(a.path, a.target, plan.link_mode)
                else:
                    for p in [a.path, *a.extra_paths]:
                        with contextlib.suppress(FileNotFoundError):
                            p.unlink()
                entry["ok"] = True
            except (OSError, ValueError) as e:
                entry["ok"] = False
                entry["error"] = str(e)
            log.write(json.dumps(entry) + "\n")
            results.append(entry)
    return results
