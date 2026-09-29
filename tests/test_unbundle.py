from __future__ import annotations

from helpers import add_ollama_model, write_gguf

from llm_doctor.stores import ollama
from llm_doctor.unbundle import plan_exports, render_llama_swap, write_exports


def models(env, tmp_path):
    root = env["ollama"]
    add_ollama_model(
        root,
        "qwen3",
        write_gguf(tmp_path / "q.gguf").read_bytes(),
        tag="1.7b",
        params={
            "num_ctx": 8192,
            "temperature": 0.6,
            "top_k": 20,
            "stop": ["<|im_end|>"],
            "mystery": 1,
        },
        system="You are terse.",
    )
    add_ollama_model(
        root,
        "gemma3",
        write_gguf(tmp_path / "g.gguf", arch="gemma3", seed=4).read_bytes(),
        tag="4b",
        projector=write_gguf(tmp_path / "p.gguf", mmproj=True).read_bytes(),
    )
    return ollama.scan(root).models


def test_llama_server_presets(env, tmp_path):
    out = tmp_path / "out"
    exports = plan_exports(models(env, tmp_path), "llama-server", out, 32768)
    written = write_exports(exports, "llama-server", out)
    assert not written["errors"]
    ini = (out / "presets.ini").read_text()
    assert "version = 1" in ini and "[*]" in ini
    assert "[qwen3:1.7b]" in ini
    assert f"model = {out / 'qwen3-1.7b.gguf'}" in ini
    assert "ctx-size = 8192" in ini and "temp = 0.6" in ini and "top-k = 20" in ini
    assert '["<|im_end|>"]' in ini
    assert "You are terse." in ini
    # vision model goes in its own folder with an mmproj file, as --models-dir expects
    assert (out / "gemma3-4b" / "gemma3-4b.gguf").is_symlink()
    assert (out / "gemma3-4b" / "mmproj-gemma3-4b.gguf").is_symlink()
    assert f"mmproj = {out / 'gemma3-4b' / 'mmproj-gemma3-4b.gguf'}" in ini
    # no num_ctx in the Modelfile: min(trained context, --ctx)
    assert "ctx-size = 32768" in ini
    q = next(e for e in exports if e.name == "qwen3:1.7b")
    assert any("mystery" in n for n in q.notes)
    # running again is idempotent
    again = write_exports(exports, "llama-server", out)
    assert {link["mode"] for link in again["links"]} == {"exists"}


def test_llama_swap_and_hardlink(env, tmp_path):
    out = tmp_path / "swap"
    exports = plan_exports(models(env, tmp_path), "llama-swap", out, 16384, only=["qwen3:1.7b"])
    assert [e.name for e in exports] == ["qwen3:1.7b"]
    write_exports(exports, "llama-swap", out, link_mode="hardlink", binary="/opt/llama-server")
    yml = (out / "llama-swap.yaml").read_text()
    assert yml.startswith("# llama-swap config")
    assert '"qwen3:1.7b":' in yml
    assert "/opt/llama-server --port ${PORT}" in yml
    assert "--ctx-size 8192" in yml and "--jinja" in yml
    assert '"stop?": ["<|im_end|>"]' in yml
    assert not (out / "qwen3-1.7b.gguf").is_symlink()
    assert (
        render_llama_swap([], "x").strip()
        == "# llama-swap config written by llm-doctor unbundle\n# Start with: llama-swap --config <this file>\nmodels:".strip()
    )


def test_lmstudio_and_mlx(env, tmp_path):
    ms = models(env, tmp_path)
    out = tmp_path / "lms"
    exports = plan_exports(ms, "lmstudio", out, 32768)
    write_exports(exports, "lmstudio", out)
    assert (out / "qwen3-1.7b" / "qwen3-1.7b.gguf").is_symlink()
    mlx_out = tmp_path / "mlx"
    write_exports(plan_exports(ms, "mlx", mlx_out, 32768), "mlx", mlx_out)
    script = (mlx_out / "convert-to-mlx.sh").read_text()
    assert "source repo unknown" in script


def test_ollama_engine_model_with_embedded_vision(env, tmp_path):
    g = write_gguf(tmp_path / "o.gguf", arch="glmocr", template=None, embedded_vision=True)
    add_ollama_model(
        env["ollama"],
        "glm-ocr",
        g.read_bytes(),
        tag="q8_0",
        with_template=False,
        config_extra={"renderer": "glm-ocr", "parser": "glm-ocr", "requires": "0.15.5"},
    )
    m = ollama.scan(env["ollama"]).models[0]
    assert m.meta["embedded_vision"] is True and m.meta["renderer"] == "glm-ocr"
    out = tmp_path / "out"
    exports = plan_exports([m], "llama-server", out, 32768)
    assert len(exports[0].warnings) == 2
    write_exports(exports, "llama-server", out)
    ini = (out / "presets.ini").read_text()
    assert "; warning: vision weights are inside this GGUF" in ini
    assert "'glm-ocr' renderer" in ini
