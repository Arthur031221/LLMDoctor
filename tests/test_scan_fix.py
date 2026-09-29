from __future__ import annotations

import json
import os

import pytest
from helpers import add_hf_file, add_ollama_blob, add_ollama_model, age, write_gguf

from llm_doctor.fix import apply_fix, plan_fix
from llm_doctor.scan import ScanOptions, run_scan


@pytest.fixture
def world(env, tmp_path):
    """Ollama and LM Studio hold the same GGUF, plus an orphan and a stale partial."""
    g = write_gguf(tmp_path / "shared.gguf", tensor_floats=64_000).read_bytes()
    add_ollama_model(
        env["ollama"], "Qwen3-0.6B-GGUF", g, host="hf.co", namespace="unsloth", tag="Q4_K_M"
    )
    lm = env["lmstudio"] / "unsloth" / "Qwen3-0.6B-GGUF" / "Qwen3-0.6B-Q4_K_M.gguf"
    lm.parent.mkdir(parents=True)
    lm.write_bytes(g)
    orphan = add_ollama_blob(env["ollama"], b"o" * 5000)
    orphan_path = env["ollama"] / "blobs" / f"sha256-{orphan}"
    age(orphan_path, 7200)
    partial = env["ollama"] / "blobs" / f"sha256-{'a' * 64}-partial"
    partial.write_bytes(b"p" * 3000)
    age(partial, 7200)
    fresh = env["ollama"] / "blobs" / f"sha256-{'b' * 64}-partial"
    fresh.write_bytes(b"q" * 10)
    return {**env, "gguf": g, "lm": lm, "orphan": orphan_path, "partial": partial, "fresh": fresh}


def scan(**kw):
    return run_scan(ScanOptions(online=False, min_size=1024, ram=24 * 1024**3, **kw))


def test_scan_finds_duplicates_leftovers_and_fits(world):
    report = scan()
    assert [s.name for s in report.stores] == ["ollama", "lmstudio"]
    assert len(report.duplicates) == 1
    g = report.duplicates[0]
    assert {c.store for c in g.copies} == {"ollama", "lmstudio"}
    assert g.keeper().store == "ollama"
    assert g.reclaimable == len(world["gguf"])
    codes = [f.code for f in report.findings]
    assert codes.count("duplicate") == 1
    assert "orphan" in codes and "incomplete" in codes
    # only the LM Studio copy needed hashing, the Ollama blob is named by its sha256
    assert report.hashed_bytes == len(world["gguf"])
    assert all(report.fits[k].status == "fits" for k in report.fits)
    data = report.to_dict()
    json.dumps(data)
    assert data["reclaimable_bytes"] == len(world["gguf"]) + 5000 + 3000
    # second scan uses the hash cache
    assert scan().hashed_bytes == 0


def test_fix_dry_run_then_apply(world):
    report = scan()
    plan = plan_fix(report)
    kinds = sorted((a.category, a.kind) for a in plan.actions)
    assert kinds == [("dedupe", "link"), ("incomplete", "delete"), ("orphans", "delete")]
    assert any("younger than --min-age" in s.reason for s in plan.skipped)
    assert world["lm"].is_file() and not world["lm"].is_symlink()

    results = apply_fix(plan)
    assert all(r["ok"] for r in results)
    assert world["lm"].is_symlink()
    assert world["lm"].read_bytes() == world["gguf"]
    assert not world["orphan"].exists() and not world["partial"].exists()
    assert world["fresh"].exists()
    log = (world["home"] / ".llm-doctor" / "fix-log.jsonl").read_text().splitlines()
    assert len(log) == 3

    after = scan()
    assert after.duplicates == []
    assert after.stores[1].disk_bytes() == 0  # the LM Studio copy is now a link


def test_fix_hardlink_and_only(world):
    plan = plan_fix(scan(), link_mode="hardlink", only={"dedupe"})
    assert [a.category for a in plan.actions] == ["dedupe"]
    apply_fix(plan)
    assert not world["lm"].is_symlink()
    blob = next(
        p
        for p in (world["ollama"] / "blobs").glob("sha256-*")
        if p.stat().st_size == len(world["gguf"])
    )
    assert (
        os.stat(world["lm"]).st_nlink == 2 and os.stat(blob).st_ino == os.stat(world["lm"]).st_ino
    )


def test_fix_never_rewrites_hf_cache(env, tmp_path):
    data = write_gguf(tmp_path / "x.gguf", tensor_floats=10_000).read_bytes()
    add_hf_file(env["hf"], "a/b-GGUF", "m.gguf", data)
    add_hf_file(env["hf"], "c/d-GGUF", "m.gguf", data)
    report = scan()
    assert len(report.duplicates) == 1
    plan = plan_fix(report)
    assert plan.actions == []
    assert any("huggingface_hub" in s.reason for s in plan.skipped)


def test_empty_machine(env):
    report = scan()
    assert report.stores == [] and report.findings == []
    assert plan_fix(report).actions == []


def test_memory_findings(env, tmp_path):
    big = write_gguf(
        tmp_path / "big.gguf", layers=80, kv_heads=8, head_dim=128, ctx=131072
    ).read_bytes()
    add_ollama_model(env["ollama"], "big", big)
    report = run_scan(ScanOptions(online=False, ctx=131072, ram=24 * 1024**3))
    f = next(f for f in report.findings if f.code in ("no-fit", "tight-fit"))
    # 80 layers * 8 * 256 * 2 bytes * 131072 tokens = 40 GiB of KV cache
    assert f.code == "no-fit" and "KV cache" in f.message
