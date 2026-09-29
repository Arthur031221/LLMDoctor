from __future__ import annotations

from dataclasses import asdict, dataclass, field

SEVERITY_ORDER = {"error": 0, "warn": 1, "info": 2}


@dataclass
class Finding:
    severity: str  # error, warn, info
    code: str
    subject: str
    message: str
    store: str | None = None
    fix: str | None = None
    bytes: int = 0
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def sort_findings(findings: list[Finding]) -> list[Finding]:
    return sorted(findings, key=lambda f: (SEVERITY_ORDER.get(f.severity, 9), f.code, f.subject))
