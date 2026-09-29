"""Data types shared by the store scanners."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from llm_doctor.memory import ArchInfo

# Files smaller than this are never considered for dedupe.
MIN_DEDUPE_BYTES = 16 * 1024 * 1024

WEIGHT_SUFFIXES = (".gguf", ".safetensors", ".bin", ".npz", ".pt", ".pth")
PARTIAL_SUFFIXES = (".part", ".partial", ".download", ".crdownload", ".incomplete")


@dataclass
class WeightFile:
    path: Path  # physical file we would hash or link
    display: str  # path shown to the user
    size: int
    role: str = "model"  # model, projector, adapter, weights
    digest: str | None = None  # sha256 known from the store layout, no hashing needed

    def identity(self) -> tuple[int, int] | None:
        try:
            st = self.path.stat()
        except OSError:
            return None
        return (st.st_dev, st.st_ino)

    def is_link(self) -> bool:
        return self.path.is_symlink()


@dataclass
class Model:
    store: str
    name: str
    format: str  # gguf, mlx, safetensors, unknown
    location: str
    files: list[WeightFile] = field(default_factory=list)
    source_repo: str | None = None  # Hugging Face repo id when known
    source_file: str | None = None  # file name inside that repo
    revision: str | None = None
    arch: ArchInfo | None = None
    chat_template: str | None = None
    template_source: str | None = None
    meta: dict = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return sum(f.size for f in self.files)

    @property
    def weight_bytes(self) -> int:
        return sum(f.size for f in self.files if f.role in ("model", "weights"))


@dataclass
class Leftover:
    store: str
    kind: str  # incomplete, orphan
    label: str
    paths: list[Path]
    size: int
    mtime: float

    def age_seconds(self, now: float) -> float:
        return max(0.0, now - self.mtime)


@dataclass
class StoreResult:
    name: str
    root: Path
    models: list[Model] = field(default_factory=list)
    leftovers: list[Leftover] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def disk_bytes(self) -> int:
        """Unique bytes on disk for model files and leftovers, counting each inode once."""
        seen: set[tuple[int, int]] = set()
        total = 0
        paths: list[Path] = [wf.path for m in self.models for wf in m.files]
        paths += [p for lo in self.leftovers for p in lo.paths]
        for p in paths:
            try:
                st = os.stat(p)
            except OSError:
                continue
            key = (st.st_dev, st.st_ino)
            if key in seen:
                continue
            seen.add(key)
            # A symlink we created points into another store, so it costs nothing here.
            if Path(p).is_symlink():
                continue
            total += st.st_size
        return total


def file_size(p: Path) -> int:
    try:
        return p.stat().st_size
    except OSError:
        return 0


def file_mtime(p: Path) -> float:
    try:
        return p.lstat().st_mtime
    except OSError:
        return 0.0
