"""Small helpers shared across modules."""

from __future__ import annotations

import hashlib
import os
import platform
import re
import subprocess
from pathlib import Path


def human_bytes(n: float | int | None) -> str:
    if n is None:
        return "?"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1000 or unit == "TB":
            if unit == "B":
                return f"{int(n)} B"
            return f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} TB"


def human_mem(n: float | int | None) -> str:
    """Memory sizes in binary units, which is how Apple and GPU tools report RAM."""
    if n is None:
        return "?"
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024 or unit == "TiB":
            return f"{int(n)} B" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TiB"


def doctor_home() -> Path:
    """Directory for llm-doctor's own state (hash cache, fix log)."""
    env = os.environ.get("LLM_DOCTOR_HOME")
    return Path(env).expanduser() if env else Path.home() / ".llm-doctor"


def short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def normalize_template(text: str) -> str:
    """Normalize a Jinja chat template so cosmetic whitespace does not count as a change."""
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").split("\n")]
    return "\n".join(lines).strip()


def template_hash(text: str | None) -> str | None:
    if not text:
        return None
    return short_hash(normalize_template(text))


def _sysctl(name: str) -> str | None:
    try:
        out = subprocess.run(
            ["/usr/sbin/sysctl", "-n", name], capture_output=True, text=True, timeout=5
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def physical_ram() -> int | None:
    if platform.system() == "Darwin":
        v = _sysctl("hw.memsize")
        return int(v) if v and v.isdigit() else None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def gpu_wired_limit(ram: int | None) -> tuple[int | None, str]:
    """Return the usable GPU memory budget on Apple Silicon and where the number came from.

    macOS caps how much unified memory the GPU may wire. The cap can be raised with
    `sudo sysctl iogpu.wired_limit_mb=N`. When it is unset (0), macOS uses a default of
    about two thirds of RAM on machines with 36 GB or less and three quarters above that.
    """
    if platform.system() != "Darwin" or ram is None:
        return ram, "physical RAM"
    v = _sysctl("iogpu.wired_limit_mb")
    if v and v.isdigit() and int(v) > 0:
        return int(v) * 1024 * 1024, "iogpu.wired_limit_mb"
    frac = 0.75 if ram > 36 * 1024**3 else 2 / 3
    return int(ram * frac), "macOS default GPU limit (approx.)"


def available_memory() -> int | None:
    """Memory the OS could hand to a new process without swapping, best effort."""
    if platform.system() == "Darwin":
        try:
            out = subprocess.run(["/usr/bin/vm_stat"], capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.SubprocessError):
            return None
        if out.returncode != 0:
            return None
        page = 16384
        m = re.search(r"page size of (\d+) bytes", out.stdout)
        if m:
            page = int(m.group(1))
        pages = {}
        for line in out.stdout.splitlines():
            if ":" in line:
                k, _, v = line.partition(":")
                v = v.strip().rstrip(".")
                if v.isdigit():
                    pages[k.strip()] = int(v)
        free = sum(
            pages.get(k, 0)
            for k in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
        )
        return free * page
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        return None
    return None


def is_local_host(host: str | None) -> bool:
    return (host or "").lower() in {"localhost", "127.0.0.1", "::1", "0.0.0.0", "[::1]"}


def slugify(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-.")
    return re.sub(r"-{2,}", "-", s) or "model"
