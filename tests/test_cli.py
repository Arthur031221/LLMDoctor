from __future__ import annotations

import json

from helpers import add_ollama_model, write_gguf
from typer.testing import CliRunner

from llm_doctor import __version__
from llm_doctor.cli import app

runner = CliRunner()


def test_version_and_help():
    r = runner.invoke(app, ["--version"])
    assert r.exit_code == 0 and __version__ in r.output
    for cmd in ([], ["scan"], ["fix"], ["unbundle"], ["endpoint"]):
        r = runner.invoke(app, [*cmd, "--help"])
        assert r.exit_code == 0, cmd
        assert "Usage" in r.output


def test_scan_empty_machine(env):
    r = runner.invoke(app, ["scan", "--offline"])
    assert r.exit_code == 0
    assert "No model stores found" in r.output
    r = runner.invoke(app, ["--json", "--offline"])
    assert json.loads(r.output)["stores"] == []


def test_scan_fix_unbundle_json(env, tmp_path):
    g = write_gguf(tmp_path / "m.gguf", tensor_floats=100_000).read_bytes()
    add_ollama_model(env["ollama"], "qwen3", g, tag="1.7b", params={"num_ctx": 4096})
    lm = env["lmstudio"] / "Qwen" / "Qwen3-GGUF" / "q.gguf"
    lm.parent.mkdir(parents=True)
    lm.write_bytes(g)

    r = runner.invoke(app, ["scan", "--offline", "--json", "--min-size", "0"])
    data = json.loads(r.output)
    assert len(data["duplicates"]) == 1
    assert {m["store"] for m in data["models"]} == {"ollama", "lmstudio"}

    r = runner.invoke(app, ["scan", "--offline", "--min-size", "0"])
    assert "duplicate" in r.output and "reclaimable" in r.output

    r = runner.invoke(app, ["fix", "--json", "--min-size", "0"])
    plan = json.loads(r.output)
    assert plan["applied"] is False and plan["reclaim_bytes"] == len(g)
    assert not lm.is_symlink()
    r = runner.invoke(app, ["fix", "--yes", "--min-size", "0"])
    assert r.exit_code == 0 and "Reclaimed" in r.output
    assert lm.is_symlink()

    r = runner.invoke(app, ["fix", "--only", "nonsense"])
    assert r.exit_code == 2

    out = tmp_path / "out"
    r = runner.invoke(app, ["unbundle", "--out", str(out), "--dry-run", "--json"])
    assert json.loads(r.output)["models"][0]["context"] == 4096
    assert not out.exists()
    r = runner.invoke(app, ["unbundle", "--to", "llama-swap", "--out", str(out)])
    assert r.exit_code == 0 and (out / "llama-swap.yaml").exists()
    r = runner.invoke(app, ["unbundle", "--to", "vllm"])
    assert r.exit_code == 2
    r = runner.invoke(app, ["unbundle", "--model", "nope"])
    assert r.exit_code == 1


def test_endpoint_bad_args_and_unreachable():
    assert runner.invoke(app, ["endpoint", "http://x", "--api", "grpc"]).exit_code == 2
    assert runner.invoke(app, ["endpoint", "http://x", "--only", "vibes"]).exit_code == 2
    r = runner.invoke(app, ["endpoint", "127.0.0.1:9", "--json", "--budget", "10"])
    assert r.exit_code == 1
    assert json.loads(r.output)["failed"] is True
