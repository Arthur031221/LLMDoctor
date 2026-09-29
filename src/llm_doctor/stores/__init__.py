"""Store discovery. Each store scanner returns a StoreResult."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from llm_doctor.stores import folders, huggingface, ollama
from llm_doctor.stores.base import Leftover, Model, StoreResult, WeightFile

__all__ = ["Leftover", "Model", "StoreLocation", "StoreResult", "WeightFile", "discover", "scan"]


@dataclass(frozen=True)
class StoreLocation:
    name: str
    root: Path

    def to_dict(self) -> dict:
        return {"store": self.name, "root": str(self.root), "exists": self.root.is_dir()}


def discover(
    extra_paths: list[Path] | None = None, include_missing: bool = False
) -> list[StoreLocation]:
    locs: list[StoreLocation] = []
    for root in ollama.default_roots():
        locs.append(StoreLocation("ollama", root))
    for root in folders.lmstudio_roots():
        locs.append(StoreLocation("lmstudio", root))
    for root in huggingface.default_roots():
        locs.append(StoreLocation("huggingface", root))
    for root in folders.mlx_roots():
        locs.append(StoreLocation("mlx", root))
    for root in [*folders.extra_roots_from_env(), *(extra_paths or [])]:
        locs.append(StoreLocation("path", Path(root).expanduser()))
    seen: set[Path] = set()
    out = []
    for loc in locs:
        key = loc.root.resolve() if loc.root.exists() else loc.root
        if key in seen:
            continue
        seen.add(key)
        if include_missing or loc.root.is_dir():
            out.append(loc)
    return out


def scan(loc: StoreLocation) -> StoreResult:
    if loc.name == "ollama":
        return ollama.scan(loc.root)
    if loc.name == "huggingface":
        return huggingface.scan(loc.root)
    return folders.scan(loc.root, loc.name)
