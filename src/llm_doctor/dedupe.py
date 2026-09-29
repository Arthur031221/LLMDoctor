"""Find byte-identical weight files across stores.

Files are grouped by size first. Only files that share a size with another physical file
are hashed, and only when the store does not already name them by sha256 (Ollama blobs
and Hugging Face LFS blobs do). Hashes are cached in ~/.llm-doctor/cache.json.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from llm_doctor.hashcache import HashCache
from llm_doctor.stores.base import MIN_DEDUPE_BYTES, Model, WeightFile

# Which copy to keep when a file exists in several stores. The Hugging Face cache comes
# first because its blobs are read-only and reference counted by huggingface_hub, so
# rewriting them would confuse its own cleanup. Ollama is next because it resolves
# models through content-addressed blob names that other tools cannot see.
KEEP_PRIORITY = {"huggingface": 0, "ollama": 1, "lmstudio": 2, "mlx": 3, "path": 4}


@dataclass
class Copy:
    store: str
    models: list[str]
    file: WeightFile

    def to_dict(self) -> dict:
        return {
            "store": self.store,
            "models": self.models,
            "path": self.file.display,
            "physical_path": str(self.file.path),
        }


@dataclass
class DupGroup:
    sha256: str
    size: int
    copies: list[Copy] = field(default_factory=list)

    @property
    def reclaimable(self) -> int:
        return self.size * (len(self.copies) - 1)

    def keeper(self) -> Copy:
        return min(
            self.copies,
            key=lambda c: (KEEP_PRIORITY.get(c.store, 9), c.file.is_link(), _mtime(c.file)),
        )

    def to_dict(self) -> dict:
        return {
            "sha256": self.sha256,
            "size": self.size,
            "reclaimable": self.reclaimable,
            "keep": self.keeper().file.display,
            "copies": [c.to_dict() for c in self.copies],
        }


def _mtime(wf: WeightFile) -> float:
    try:
        return wf.path.stat().st_mtime
    except OSError:
        return 0.0


def find_duplicates(
    models: list[Model],
    cache: HashCache,
    min_size: int = MIN_DEDUPE_BYTES,
    progress: Callable[[int], None] | None = None,
    on_hash_start: Callable[[int], None] | None = None,
) -> list[DupGroup]:
    # Physical identity -> list of (model, file). Several Ollama tags can share one blob.
    by_ident: dict[tuple[int, int], list[tuple[Model, WeightFile]]] = {}
    for m in models:
        for wf in m.files:
            if wf.size < min_size:
                continue
            ident = wf.identity()
            if ident is None:
                continue
            by_ident.setdefault(ident, []).append((m, wf))

    by_size: dict[int, list[tuple[int, int]]] = {}
    for ident, refs in by_ident.items():
        by_size.setdefault(refs[0][1].size, []).append(ident)

    candidates = [idents for idents in by_size.values() if len(idents) > 1]
    to_hash = 0
    for idents in candidates:
        for ident in idents:
            refs = by_ident[ident]
            if not any(wf.digest for _, wf in refs) and not cache.lookup(refs[0][1].path):
                to_hash += refs[0][1].size
    if on_hash_start and to_hash:
        on_hash_start(to_hash)

    groups: dict[str, DupGroup] = {}
    for idents in candidates:
        for ident in idents:
            refs = by_ident[ident]
            digest = next((wf.digest for _, wf in refs if wf.digest), None)
            rep = min(refs, key=lambda r: r[1].is_link())[1]
            if digest is None:
                try:
                    digest = cache.sha256(rep.path, progress)
                except OSError:
                    continue
            g = groups.setdefault(digest, DupGroup(sha256=digest, size=rep.size))
            names = sorted({m.name for m, _ in refs})
            g.copies.append(Copy(store=refs[0][0].store, models=names, file=rep))
    out = [g for g in groups.values() if len(g.copies) > 1]
    out.sort(key=lambda g: -g.reclaimable)
    return out
