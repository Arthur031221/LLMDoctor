from __future__ import annotations

import hashlib
import json

import httpx
from helpers import TEMPLATE_NEW, TEMPLATE_OLD, add_hf_file, add_ollama_model, write_gguf

from llm_doctor.remote import Remote
from llm_doctor.scan import ScanOptions, run_scan


def make_remote(routes: dict) -> Remote:
    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url).split("?")[0]
        for key, value in routes.items():
            if url.endswith(key):
                if callable(value):
                    return value(request)
                if isinstance(value, bytes):
                    return httpx.Response(206, content=value)
                return httpx.Response(200, json=value)
        return httpx.Response(404, json={"error": "not found"})

    return Remote(transport=httpx.MockTransport(handler))


def test_hf_gguf_outdated_and_stale_template(env, tmp_path):
    old = write_gguf(
        tmp_path / "old.gguf", template=TEMPLATE_OLD, base_repo="Qwen/Qwen3-0.6B"
    ).read_bytes()
    new = write_gguf(tmp_path / "new.gguf", template=TEMPLATE_NEW, seed=9).read_bytes()
    add_hf_file(
        env["hf"], "unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf", old, revision="1" * 40
    )
    remote = make_remote(
        {
            "/api/models/unsloth/Qwen3-0.6B-GGUF": {
                "sha": "2" * 40,
                "gguf": {"chat_template": TEMPLATE_NEW},
            },
            "/paths-info/main": [
                {
                    "path": "Qwen3-0.6B-Q4_K_M.gguf",
                    "size": len(new),
                    "lfs": {"oid": hashlib.sha256(new).hexdigest(), "size": len(new)},
                }
            ],
            "/resolve/main/Qwen3-0.6B-Q4_K_M.gguf": new,
            "Qwen/Qwen3-0.6B/resolve/main/tokenizer_config.json": {"chat_template": "base {{ x }}"},
        }
    )
    report = run_scan(ScanOptions(online=True), remote=remote)
    codes = {f.code: f for f in report.findings}
    assert "outdated" in codes
    assert codes["template-stale"].data["lines_added"] >= 1
    assert "template-vs-base" in codes
    assert codes["outdated"].fix.startswith("hf download unsloth/Qwen3-0.6B-GGUF")


def test_hf_gguf_current_is_quiet(env, tmp_path):
    data = write_gguf(tmp_path / "m.gguf", template=TEMPLATE_NEW).read_bytes()
    add_hf_file(env["hf"], "o/r-GGUF", "m.gguf", data)
    remote = make_remote(
        {
            "/api/models/o/r-GGUF": {"sha": "a" * 40},
            "/paths-info/main": [
                {
                    "path": "m.gguf",
                    "size": len(data),
                    "lfs": {"oid": hashlib.sha256(data).hexdigest()},
                }
            ],
        }
    )
    report = run_scan(ScanOptions(online=True), remote=remote)
    assert [f.code for f in report.findings] == []


def test_ollama_registry_template_update(env, tmp_path):
    g = write_gguf(tmp_path / "m.gguf").read_bytes()
    info = add_ollama_model(env["ollama"], "qwen3", g, tag="1.7b")
    newer = json.loads(json.dumps(info["manifest"]))
    tpl = next(ly for ly in newer["layers"] if ly["mediaType"].endswith("template"))
    tpl["digest"] = "sha256:" + "9" * 64
    tpl["size"] = 1500

    def registry(request: httpx.Request) -> httpx.Response:
        assert request.headers["accept"].startswith("application/vnd.docker")
        return httpx.Response(200, content=json.dumps(newer).encode())

    remote = make_remote({"/v2/library/qwen3/manifests/1.7b": registry})
    report = run_scan(ScanOptions(online=True), remote=remote)
    out = {f.code: f for f in report.findings}
    assert out["outdated"].data["changed_layers"] == ["template"]
    assert out["outdated"].data["template_only"] is True
    assert out["outdated"].fix == "ollama pull qwen3:1.7b"
    assert "template-stale" in out


def test_network_errors_do_not_break_scan(env, tmp_path):
    add_ollama_model(env["ollama"], "qwen3", write_gguf(tmp_path / "m.gguf").read_bytes())

    def boom(request):
        raise httpx.ConnectError("offline")

    remote = Remote(transport=httpx.MockTransport(boom))
    report = run_scan(ScanOptions(online=True), remote=remote)
    assert report.findings == []
    assert report.network_errors
