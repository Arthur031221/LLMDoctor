"""Terminal output. JSON output lives next to each data type (to_dict)."""

from __future__ import annotations

from pathlib import Path

from rich.console import Console
from rich.table import Table
from rich.text import Text

from llm_doctor.findings import Finding
from llm_doctor.util import human_bytes, human_mem

SEV_STYLE = {"error": "bold red", "warn": "yellow", "info": "cyan"}
SEV_LABEL = {"error": "ERROR", "warn": "WARN", "info": "INFO"}
FIT_STYLE = {"fits": "green", "tight": "yellow", "no-fit": "bold red", "unknown": "dim"}


def tilde(path: str | Path) -> str:
    s = str(path)
    home = str(Path.home())
    return "~" + s[len(home) :] if s.startswith(home) else s


def table(*headers: str, **kw) -> Table:
    t = Table(box=None, show_edge=False, pad_edge=False, header_style="bold", **kw)
    for h in headers:
        t.add_column(h, overflow="fold")
    return t


def render_findings(console: Console, findings: list[Finding], show_info: bool = True) -> None:
    shown = [f for f in findings if show_info or f.severity != "info"]
    if not shown:
        console.print(Text("No problems found.", style="green"))
        return
    for f in shown:
        line = Text()
        line.append(f"{SEV_LABEL[f.severity]:<5} ", style=SEV_STYLE[f.severity])
        line.append(f"{f.code:<17}", style="bold")
        line.append(" ")
        line.append(f.subject)
        console.print(line, soft_wrap=True)
        detail = f.message
        if f.bytes and f.code in ("duplicate", "orphan", "incomplete", "old-revision"):
            detail += f" [{human_bytes(f.bytes)}]"
        console.print(Text(f"      {detail}"), soft_wrap=True)
        if f.fix:
            console.print(Text(f"      fix: {f.fix}", style="dim"), soft_wrap=True)


def render_scan(console: Console, report, verbose: bool = False) -> None:
    from llm_doctor.scan import model_key

    models = report.models
    console.print(
        Text.assemble(
            ("llm-doctor scan", "bold"),
            f"  {len(report.stores)} stores, {len(models)} models, "
            f"{human_bytes(report.total_bytes())} on disk ({report.elapsed:.1f} s)",
        )
    )
    if not report.stores:
        console.print(
            "No model stores found. Looked for Ollama, LM Studio, the Hugging Face cache and MLX "
            "folders. Pass --path DIR to scan a folder of GGUF or MLX models."
        )
        return
    console.print()
    t = table("store", "root", "models", "size")
    for s in report.stores:
        t.add_row(s.name, tilde(s.root), str(len(s.models)), human_bytes(s.disk_bytes()))
    console.print(t)

    if models:
        console.print()
        budget = (
            f"GPU budget {human_mem(report.budget)} of {human_mem(report.ram)} RAM"
            if report.ram
            else "RAM unknown"
        )
        console.print(
            Text(f"Models, memory at {report.ctx:,} tokens of context ({budget})", style="bold")
        )
        t = table("store", "model", "format", "size", "weights+KV", "fit")
        for m in sorted(models, key=lambda m: (m.store, m.name)):
            fit = report.fits.get(model_key(m))
            need = human_mem(fit.total) if fit else "?"
            status = fit.status if fit else "unknown"
            t.add_row(
                m.store,
                m.name,
                m.format,
                human_bytes(m.size),
                need,
                Text(status, style=FIT_STYLE.get(status, "")),
            )
        console.print(t)

    console.print()
    findings = report.findings if verbose else [f for f in report.findings if f.severity != "info"]
    hidden = len(report.findings) - len(findings)
    console.print(Text("Findings", style="bold"))
    render_findings(console, findings)
    if hidden:
        console.print(
            Text(f"      {hidden} info notes hidden, use --verbose to show them", style="dim")
        )

    errors = sum(1 for f in report.findings if f.severity == "error")
    warns = sum(1 for f in report.findings if f.severity == "warn")
    console.print()
    summary = Text()
    summary.append(f"{errors} errors, {warns} warnings. ", style="bold")
    rec = report.reclaimable()
    if rec:
        summary.append(f"{human_bytes(rec)} reclaimable, preview with `llm-doctor fix`.")
    console.print(summary)
    if not report.online:
        console.print(Text("Upstream checks skipped (--offline).", style="dim"))
    elif report.network_errors:
        console.print(
            Text(
                f"{len(report.network_errors)} upstream lookups failed, see --json for details.",
                style="dim",
            )
        )


def render_fix(console: Console, plan, applied: bool, results: list[dict] | None) -> None:
    if not plan.actions:
        console.print(Text("Nothing to fix.", style="green"))
    else:
        verb = "Applied" if applied else "Plan (dry run)"
        console.print(Text(f"{verb}: {len(plan.actions)} actions", style="bold"))
        t = table("action", "store", "path", "size", "detail")
        for a in plan.actions:
            what = f"{a.kind} -> {tilde(a.target)}" if a.target else a.kind
            if a.kind == "link":
                what = f"replace with {plan.link_mode} to {tilde(a.target)}"
            elif a.extra_paths:
                what = f"delete (+{len(a.extra_paths)} files)"
            t.add_row(a.category, a.store, tilde(a.path), human_bytes(a.bytes), what)
        console.print(t)
    if plan.skipped:
        console.print()
        console.print(Text("Skipped", style="bold"))
        for s in plan.skipped:
            console.print(Text(f"  {tilde(s.path)}: {s.reason}"), soft_wrap=True)
    console.print()
    if results is not None:
        ok = [r for r in results if r.get("ok")]
        failed = [r for r in results if not r.get("ok")]
        freed = sum(r["bytes"] for r in ok)
        console.print(Text(f"Reclaimed {human_bytes(freed)}.", style="bold green"))
        for r in failed:
            console.print(Text(f"  failed {tilde(r['path'])}: {r.get('error')}", style="red"))
        console.print(Text("Every change is logged in ~/.llm-doctor/fix-log.jsonl.", style="dim"))
    elif plan.actions:
        console.print(
            Text(
                f"Would reclaim {human_bytes(plan.reclaim_bytes)}. Nothing changed. "
                "Run `llm-doctor fix --yes` to apply.",
                style="bold",
            )
        )


def render_unbundle(
    console: Console, exports, target: str, out: Path, written, dry_run: bool
) -> None:
    if not exports:
        console.print("No Ollama models found. Pull one with `ollama pull <model>` first.")
        return
    console.print(Text(f"unbundle to {target}: {tilde(out)}", style="bold"))
    t = table("model", "file", "ctx", "notes")
    for ex in exports:
        if ex.skipped:
            t.add_row(ex.name, Text("skipped", style="yellow"), "", ex.skipped)
            continue
        f = tilde(ex.out_model)
        if ex.out_projector:
            f += f"\n{tilde(ex.out_projector)}"
        t.add_row(ex.name, f, f"{ex.ctx:,}" if ex.ctx else "", "\n".join(ex.notes))
    console.print(t)
    console.print()
    if dry_run:
        console.print("Dry run, nothing written.")
        return
    if written:
        for c in written["configs"]:
            console.print(f"wrote {tilde(c)}")
        for e in written["errors"]:
            console.print(Text(f"error: {e}", style="red"))
    if target == "llama-server":
        console.print(f"Run: llama-server --models-preset {tilde(out / 'presets.ini')}")
    elif target == "llama-swap":
        console.print(f"Run: llama-swap --config {tilde(out / 'llama-swap.yaml')}")
    elif target == "lmstudio":
        console.print(
            "LM Studio picks the models up on its next scan. Set each context length in its load settings."
        )
    elif target == "mlx":
        console.print(f"Run: sh {tilde(out / 'convert-to-mlx.sh')}")


STATUS_STYLE = {
    "PASS": "green",
    "WARN": "yellow",
    "FAIL": "bold red",
    "SKIP": "dim",
    "INFO": "cyan",
}


def render_endpoint(console: Console, report) -> None:
    b = report.backend
    who = b.kind + (f" {b.version}" if b.version else "")
    console.print(
        Text.assemble(
            ("llm-doctor endpoint ", "bold"),
            f"{report.url}  ({who}, model {b.model or '?'}, {report.api} API, {report.elapsed:.0f} s)",
        )
    )
    head = report.headline()
    if head:
        c = report.check("context")
        console.print(Text(head, style=STATUS_STYLE.get(c.status if c else "", "bold")))
    console.print()
    t = table("check", "result", "detail")
    for c in report.checks:
        t.add_row(c.name, Text(c.status, style=STATUS_STYLE.get(c.status, "")), c.detail)
    console.print(t)
    if report.fixes:
        console.print()
        console.print(Text(f"Fixes for {b.kind}", style="bold"))
        for i, f in enumerate(report.fixes, 1):
            console.print(Text(f"{i}. {f}"), soft_wrap=True)
