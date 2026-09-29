"""Command line interface."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TextColumn, TransferSpeedColumn

from llm_doctor import __version__
from llm_doctor.scan import DEFAULT_CTX, ScanOptions, run_scan

app = typer.Typer(
    name="llm-doctor",
    help="Health checks for local LLM setups: model stores, duplicates, templates, endpoints.",
    add_completion=False,
    no_args_is_help=False,
    rich_markup_mode=None,
    context_settings={"help_option_names": ["-h", "--help"]},
)

# Piped output gets a wide fixed width so tables do not fold at 80 columns.
out = Console() if sys.stdout.isatty() else Console(width=160)
err = Console(stderr=True)

PathsOpt = Annotated[
    list[Path] | None,
    typer.Option("--path", "-p", help="Extra folder with GGUF or MLX models. Repeatable."),
]
JsonOpt = Annotated[bool, typer.Option("--json", help="Print machine-readable JSON.")]
CtxOpt = Annotated[
    int, typer.Option("--ctx", help="Context length used for the memory fit estimate.", min=512)
]
MinSizeOpt = Annotated[
    float,
    typer.Option(
        "--min-size",
        help="Ignore files smaller than this many MB when looking for duplicates.",
        min=0,
    ),
]
OfflineOpt = Annotated[
    bool, typer.Option("--offline", help="Skip registry and Hugging Face Hub lookups.")
]


def emit_json(data) -> None:
    sys.stdout.write(json.dumps(data, indent=2, default=str) + "\n")


def _version(value: bool) -> None:
    if value:
        out.print(f"llm-doctor {__version__}")
        raise typer.Exit()


@app.callback(invoke_without_command=True)
def root(
    ctx: typer.Context,
    as_json: JsonOpt = False,
    offline: OfflineOpt = False,
    version: Annotated[
        bool,
        typer.Option(
            "--version", callback=_version, is_eager=True, help="Show the version and exit."
        ),
    ] = False,
) -> None:
    """Run `llm-doctor` with no command to scan every model store."""
    if ctx.invoked_subcommand is None:
        scan(
            paths=None,
            ctx_len=DEFAULT_CTX,
            offline=offline,
            no_hash=False,
            verbose=False,
            as_json=as_json,
        )


def _hash_progress(enabled: bool):
    if not enabled:
        return None, None
    progress = Progress(
        TextColumn("hashing"),
        BarColumn(),
        DownloadColumn(),
        TransferSpeedColumn(),
        console=err,
        transient=True,
    )
    state: dict = {}

    def start(total: int) -> None:
        progress.start()
        state["task"] = progress.add_task("hash", total=total)

    def advance(n: int) -> None:
        if "task" in state:
            progress.advance(state["task"], n)

    return (start, advance), progress


@app.command()
def scan(
    paths: PathsOpt = None,
    ctx_len: CtxOpt = DEFAULT_CTX,
    offline: OfflineOpt = False,
    no_hash: Annotated[
        bool, typer.Option("--no-hash", help="Skip duplicate detection (no file hashing).")
    ] = False,
    min_size: MinSizeOpt = 16,
    verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Show info notes too.")] = False,
    as_json: JsonOpt = False,
) -> None:
    """Scan Ollama, LM Studio, the Hugging Face cache and MLX folders for problems."""
    opts = ScanOptions(
        paths=paths or [],
        ctx=ctx_len,
        online=not offline,
        hash_files=not no_hash,
        min_size=int(min_size * 1_000_000),
    )
    interactive = not as_json and err.is_terminal
    hooks, progress = _hash_progress(interactive)
    status = None
    if interactive:

        def status(msg: str) -> None:
            err.print(f"[dim]{msg}...[/dim]")

    try:
        report = run_scan(opts, status=status, hash_progress=hooks)
    finally:
        if progress is not None:
            progress.stop()
    if as_json:
        emit_json(report.to_dict())
        return
    from llm_doctor.render import render_scan

    render_scan(out, report, verbose=verbose)


@app.command()
def fix(
    yes: Annotated[
        bool, typer.Option("--yes", "-y", help="Apply the plan. Without it, nothing changes.")
    ] = False,
    hardlink: Annotated[
        bool,
        typer.Option(
            "--hardlink", help="Dedupe with hardlinks instead of symlinks (same filesystem only)."
        ),
    ] = False,
    only: Annotated[
        list[str] | None,
        typer.Option("--only", help="Limit to dedupe, orphans or incomplete. Repeatable."),
    ] = None,
    min_age: Annotated[
        int,
        typer.Option(
            "--min-age", help="Minutes a leftover must be untouched before it is deleted.", min=0
        ),
    ] = 60,
    paths: PathsOpt = None,
    min_size: MinSizeOpt = 16,
    as_json: JsonOpt = False,
) -> None:
    """Dedupe weights with links and delete orphan blobs and stale partial downloads. Dry run by default."""
    from llm_doctor.fix import KINDS, apply_fix, plan_fix
    from llm_doctor.render import render_fix

    kinds = set(only or KINDS)
    bad = kinds - set(KINDS)
    if bad:
        err.print(
            f"[red]Unknown --only value: {', '.join(sorted(bad))}. Use {', '.join(KINDS)}.[/red]"
        )
        raise typer.Exit(2)
    opts = ScanOptions(
        paths=paths or [],
        online=False,
        hash_files="dedupe" in kinds,
        min_size=int(min_size * 1_000_000),
    )
    hooks, progress = _hash_progress(not as_json and err.is_terminal)
    try:
        report = run_scan(opts, hash_progress=hooks)
    finally:
        if progress is not None:
            progress.stop()
    plan = plan_fix(report, "hardlink" if hardlink else "symlink", min_age * 60, kinds)
    results = apply_fix(plan) if yes and plan.actions else None
    if as_json:
        emit_json({**plan.to_dict(), "applied": yes, "results": results})
        return
    render_fix(out, plan, applied=yes, results=results)


@app.command()
def unbundle(
    to: Annotated[
        str, typer.Option("--to", help="llama-server, llama-swap, lmstudio or mlx.")
    ] = "llama-server",
    out_dir: Annotated[
        Path | None,
        typer.Option(
            "--out", help="Output folder. Default ./ollama-unbundled, or LM Studio's models folder."
        ),
    ] = None,
    model: Annotated[
        list[str] | None, typer.Option("--model", "-m", help="Only this Ollama model. Repeatable.")
    ] = None,
    hardlink: Annotated[
        bool, typer.Option("--hardlink", help="Hardlink instead of symlink, survives `ollama rm`.")
    ] = False,
    ctx_len: Annotated[
        int,
        typer.Option("--ctx", help="Context to write when the Modelfile sets no num_ctx.", min=512),
    ] = DEFAULT_CTX,
    llama_server: Annotated[
        str | None,
        typer.Option("--llama-server", help="Path to llama-server for llama-swap commands."),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show the plan, write nothing.")
    ] = False,
    as_json: JsonOpt = False,
) -> None:
    """Expose Ollama models as named GGUF files plus a config for another runtime. No copies."""
    from llm_doctor.render import render_unbundle
    from llm_doctor.stores import StoreLocation
    from llm_doctor.stores import ollama as ollama_store
    from llm_doctor.unbundle import TARGETS, default_out, plan_exports, write_exports

    if to not in TARGETS:
        err.print(f"[red]--to must be one of {', '.join(TARGETS)}[/red]")
        raise typer.Exit(2)
    models = []
    for root in ollama_store.default_roots():
        if root.is_dir():
            from llm_doctor.stores import scan as scan_store

            models += scan_store(StoreLocation("ollama", root)).models
    target_dir = (out_dir or default_out(to)).expanduser().resolve()
    exports = plan_exports(models, to, target_dir, ctx_len, model)
    if model and not exports:
        err.print(f"[red]No Ollama model matches {', '.join(model)}. See `ollama list`.[/red]")
        raise typer.Exit(1)
    written = None
    if exports and not dry_run:
        written = write_exports(
            exports, to, target_dir, "hardlink" if hardlink else "symlink", llama_server
        )
    if as_json:
        emit_json(
            {
                "target": to,
                "out": str(target_dir),
                "dry_run": dry_run,
                "models": [e.to_dict() for e in exports],
                "written": written,
            }
        )
        return
    render_unbundle(out, exports, to, target_dir, written, dry_run)


@app.command()
def endpoint(
    url: Annotated[
        str,
        typer.Argument(help="Server URL, e.g. http://localhost:11434 or http://localhost:8080/v1"),
    ],
    model: Annotated[
        str | None,
        typer.Option(
            "--model", "-m", help="Model to test. Default: the loaded or first listed model."
        ),
    ] = None,
    api: Annotated[
        str, typer.Option("--api", help="openai (/v1/chat/completions) or ollama (/api/chat).")
    ] = "openai",
    deep: Annotated[bool, typer.Option("--deep", help="Also test a 32k-token context.")] = False,
    budget: Annotated[
        float | None,
        typer.Option(
            "--budget",
            help="Time budget in seconds for all probes. Default 90, or 300 with --deep.",
            min=10,
        ),
    ] = None,
    only: Annotated[
        list[str] | None,
        typer.Option(
            "--only",
            help="Run only these probes: context, tools, json, think, max-tokens, prefix-cache, speed, ram.",
        ),
    ] = None,
    api_key: Annotated[
        str | None,
        typer.Option("--api-key", help="Bearer token. Also read from LLM_DOCTOR_API_KEY."),
    ] = None,
    as_json: JsonOpt = False,
) -> None:
    """Probe an OpenAI-compatible or Ollama endpoint for coding-agent readiness."""
    from llm_doctor.endpoint import run_endpoint
    from llm_doctor.endpoint.probes import PROBES
    from llm_doctor.render import render_check, render_endpoint

    if api not in ("openai", "ollama"):
        err.print("[red]--api must be openai or ollama[/red]")
        raise typer.Exit(2)
    if not url.startswith(("http://", "https://")):
        url = "http://" + url
    selected = set(only or PROBES)
    unknown = selected - set(PROBES)
    if unknown:
        err.print(
            f"[red]Unknown probe: {', '.join(sorted(unknown))}. Choose from {', '.join(PROBES)}.[/red]"
        )
        raise typer.Exit(2)
    budget = budget or (300.0 if deep else 90.0)
    live = None if as_json else (lambda c: render_check(err, c))
    if not as_json:
        err.print(f"[dim]Probing {url} (budget {budget:.0f} s)...[/dim]")
    report = run_endpoint(
        url,
        model=model,
        api=api,
        deep=deep,
        budget=budget,
        only=selected,
        api_key=api_key,
        on_check=live,
    )
    if as_json:
        emit_json(report.to_dict())
    else:
        render_endpoint(out, report)
    if any(c.status == "FAIL" for c in report.checks):
        raise typer.Exit(1)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
