"""Lazy sha256 with a persistent cache keyed by path, size, mtime and inode."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from llm_doctor.util import doctor_home

CACHE_VERSION = 1
CHUNK = 8 * 1024 * 1024


def _signature(st: os.stat_result) -> list[int]:
    return [st.st_size, st.st_mtime_ns, st.st_ino]


class HashCache:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or doctor_home() / "cache.json"
        self._entries: dict[str, dict] = {}
        self._dirty = False
        self.hashed_bytes = 0
        self._load()

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return
        if data.get("version") == CACHE_VERSION and isinstance(data.get("files"), dict):
            self._entries = data["files"]

    def lookup(self, path: Path) -> str | None:
        try:
            st = path.stat()
        except OSError:
            return None
        entry = self._entries.get(str(path.resolve()))
        if entry and entry.get("sig") == _signature(st):
            return entry.get("sha256")
        return None

    def remember(self, path: Path, sha256: str) -> None:
        try:
            st = path.stat()
        except OSError:
            return
        self._entries[str(path.resolve())] = {"sig": _signature(st), "sha256": sha256}
        self._dirty = True

    def sha256(self, path: Path, progress: Callable[[int], None] | None = None) -> str:
        cached = self.lookup(path)
        if cached:
            return cached
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while True:
                chunk = f.read(CHUNK)
                if not chunk:
                    break
                h.update(chunk)
                self.hashed_bytes += len(chunk)
                if progress:
                    progress(len(chunk))
        digest = h.hexdigest()
        self.remember(path, digest)
        return digest

    def save(self) -> None:
        if not self._dirty:
            return
        # Drop entries whose files are gone so the cache does not grow forever.
        self._entries = {k: v for k, v in self._entries.items() if os.path.exists(k)}
        tmp = None
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=".cache-", suffix=".json")
            with os.fdopen(fd, "w") as f:
                json.dump({"version": CACHE_VERSION, "files": self._entries}, f)
            os.replace(tmp, self.path)
            self._dirty = False
        except OSError:
            if tmp:
                with contextlib.suppress(OSError):
                    os.unlink(tmp)
