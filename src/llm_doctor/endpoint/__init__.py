"""`llm-doctor endpoint`: agent-readiness probe for a chat completion server."""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from urllib.parse import urlparse

import httpx

from llm_doctor.endpoint import probes
from llm_doctor.endpoint.advice import fixes
from llm_doctor.endpoint.backend import BackendInfo, detect, refresh_llama, refresh_ollama
from llm_doctor.endpoint.client import ChatClient, split_base
from llm_doctor.endpoint.probes import (
    DEEP_LEVELS,
    DEFAULT_LEVELS,
    FAIL,
    INFO,
    PASS,
    SKIP,
    WARN,
    Check,
    ProbeState,
)
from llm_doctor.util import available_memory, human_mem, is_local_host, physical_ram


@dataclass
class EndpointReport:
    url: str
    api: str
    backend: BackendInfo
    checks: list[Check] = field(default_factory=list)
    fixes: list[str] = field(default_factory=list)
    elapsed: float = 0.0

    def check(self, cid: str) -> Check | None:
        return next((c for c in self.checks if c.id == cid), None)

    def headline(self) -> str | None:
        c = self.check("context")
        if not c:
            return None
        eff, adv = c.data.get("effective"), c.data.get("advertised")
        if eff and adv:
            return f"Context: {eff:,} effective of {adv:,} advertised, {c.status}"
        if eff:
            return f"Context: {eff:,} effective, {c.status}"
        return f"Context: {c.status}"

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "api": self.api,
            "backend": self.backend.to_dict(),
            "headline": self.headline(),
            "checks": [c.to_dict() for c in self.checks],
            "fixes": self.fixes,
            "elapsed_seconds": round(self.elapsed, 1),
            "failed": any(c.status == FAIL for c in self.checks),
        }


def ram_check(info: BackendInfo, url: str) -> Check:
    host = urlparse(url).hostname
    if not is_local_host(host):
        return Check("ram", "ram headroom", SKIP, "remote endpoint, memory not visible")
    total, avail = physical_ram(), available_memory()
    if not total or avail is None:
        return Check("ram", "ram headroom", SKIP, "could not read system memory")
    resident = info.arch.get("resident_bytes") if info.arch else None
    detail = f"{human_mem(avail)} available of {human_mem(total)}"
    if resident:
        detail += f", model resident {human_mem(resident)}"
    status = PASS if avail >= 2 * 1024**3 else WARN
    if status == WARN:
        detail += ", little room for a longer context"
    return Check(
        "ram",
        "ram headroom",
        status,
        detail,
        {"available": avail, "total": total, "resident": resident},
    )


def run_endpoint(
    url: str,
    model: str | None = None,
    api: str = "openai",
    deep: bool = False,
    budget: float = 90.0,
    only: set[str] | None = None,
    api_key: str | None = None,
    on_check: Callable[[Check], None] | None = None,
    transport: httpx.BaseTransport | None = None,
) -> EndpointReport:
    t_start = time.monotonic()
    api_key = api_key or os.environ.get("LLM_DOCTOR_API_KEY")
    root, openai_base = split_base(url)
    only = only or set(probes.PROBES)
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    with httpx.Client(transport=transport, headers=headers) as http:
        info = detect(http, root, openai_base, model)
    report = EndpointReport(url=url, api=api, backend=info)
    if api == "ollama" and info.kind != "ollama":
        report.checks.append(
            Check("connect", "connect", FAIL, "--api ollama needs an Ollama server")
        )
        return report
    if not info.model:
        report.checks.append(
            Check(
                "connect", "connect", FAIL, "no model found, pass --model (is the server running?)"
            )
        )
        report.elapsed = time.monotonic() - t_start
        return report

    client = ChatClient(
        url, info.model, api=api, backend=info.kind, api_key=api_key, transport=transport
    )
    st = ProbeState(
        client=client,
        backend=info,
        deadline=t_start + budget,
        levels=DEEP_LEVELS if deep else DEFAULT_LEVELS,
        on_check=on_check,
    )
    try:
        bad = probes.calibrate(st)
        if bad:
            st.add(bad)
            report.checks = st.checks
            report.elapsed = time.monotonic() - t_start
            return report
        # The model is loaded now, so the server can say what window it allocated.
        if info.kind == "ollama":
            refresh_ollama(client.http, root, info)
        elif info.kind == "llama-server":
            refresh_llama(client.http, root, info)
        if "context" in only:
            st.add(probes.probe_context(st))
        steps = [
            ("tools", lambda: probes.probe_tools(st)),
            ("json", lambda: [probes.probe_json(st)]),
            ("think", lambda: [probes.probe_think(st)]),
            ("max-tokens", lambda: [probes.probe_max_tokens(st)]),
            ("prefix-cache", lambda: [probes.probe_prefix_cache(st)]),
            ("speed", lambda: probes.probe_speed(st)),
        ]
        for name, fn in steps:
            if name not in only:
                continue
            if st.remaining() < 3:
                st.add(Check(name, name, SKIP, "time budget used, raise --budget"))
                continue
            for c in fn():
                st.add(c)
        if "ram" in only:
            if info.kind == "ollama":
                refresh_ollama(client.http, root, info)
            st.add(ram_check(info, url))
    finally:
        client.close()
    report.checks = st.checks
    report.fixes = fixes(st.checks, info, api)
    report.elapsed = time.monotonic() - t_start
    return report


__all__ = ["FAIL", "INFO", "PASS", "SKIP", "WARN", "EndpointReport", "run_endpoint"]
