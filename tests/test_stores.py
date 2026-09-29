from __future__ import annotations

import time

from helpers import TEMPLATE_OLD, add_hf_file, add_ollama_blob, add_ollama_model, write_gguf

from llm_doctor.stores import discover, folders, huggingface, ollama


def test_ollama_scan_names_layers_orphans_partials(env, tmp_path):
    root = env["ollama"]
    g = write_gguf(tmp_path / "a.gguf").read_bytes()
    add_ollama_model(
        root,
        "qwen3",
        g,
        tag="1.7b",
        params={"num_ctx": 8192, "stop": ["<|im_end|>"]},
        system="Be brief.",
    )
    add_ollama_model(root, "Qwen3-0.6B-GGUF", g, host="hf.co", namespace="unsloth", tag="Q4_K_M")
    add_ollama_model(
        root, "tool", write_gguf(tmp_path / "b.gguf", seed=1).read_bytes(), namespace="someone"
    )
    orphan = add_ollama_blob(root, b"x" * 100)
    (root / "blobs" / f"sha256-{'f' * 64}-partial").write_bytes(b"y" * 50)
    (root / "blobs" / f"sha256-{'f' * 64}-partial-0").write_text('{"N":0}')

    res = ollama.scan(root)
    names = sorted(m.name for m in res.models)
    assert names == ["hf.co/unsloth/Qwen3-0.6B-GGUF:Q4_K_M", "qwen3:1.7b", "someone/tool:latest"]
    q = next(m for m in res.models if m.name == "qwen3:1.7b")
    assert q.format == "gguf"
    assert q.meta["params"]["num_ctx"] == 8192
    assert q.meta["system"] == "Be brief."
    assert q.chat_template == TEMPLATE_OLD
    assert q.arch.n_layers == 28
    assert q.files[0].digest and q.files[0].role == "model"
    hf = next(m for m in res.models if m.name.startswith("hf.co"))
    assert hf.source_repo == "unsloth/Qwen3-0.6B-GGUF"
    kinds = {(lo.kind, lo.label.split(" ")[0]) for lo in res.leftovers}
    assert ("orphan", f"sha256-{orphan}") in kinds
    partial = next(lo for lo in res.leftovers if lo.kind == "incomplete")
    assert len(partial.paths) == 2 and partial.size > 50
    # shared blob is counted once
    assert res.disk_bytes() < 3 * len(g) + 10_000


def test_ollama_missing_blob_and_unknown_layer(env, tmp_path):
    root = env["ollama"]
    info = add_ollama_model(root, "broken", write_gguf(tmp_path / "a.gguf").read_bytes())
    model_digest = info["manifest"]["layers"][0]["digest"].split(":")[1]
    (root / "blobs" / f"sha256-{model_digest}").unlink()
    res = ollama.scan(root)
    assert any("missing blob" in p for p in res.models[0].problems)


def test_hf_cache_xet_layout_and_revisions(env, tmp_path):
    hub = env["hf"]
    old = write_gguf(tmp_path / "old.gguf", template=TEMPLATE_OLD).read_bytes()
    new = write_gguf(tmp_path / "new.gguf", seed=3).read_bytes()
    add_hf_file(
        hub,
        "unsloth/Qwen3-0.6B-GGUF",
        "Qwen3-0.6B-Q4_K_M.gguf",
        old,
        revision="1" * 40,
        main=False,
        xet=True,
    )
    add_hf_file(
        hub, "unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf", new, revision="2" * 40, xet=True
    )
    add_hf_file(
        hub,
        "unsloth/Qwen3-0.6B-GGUF",
        "mmproj-F16.gguf",
        write_gguf(tmp_path / "p.gguf", mmproj=True).read_bytes(),
        revision="2" * 40,
    )
    inc = hub / "models--unsloth--Qwen3-0.6B-GGUF" / "blobs" / ("e" * 64 + ".incomplete")
    inc.write_bytes(b"z" * 10)

    res = huggingface.scan(hub)
    assert len(res.models) == 1
    m = res.models[0]
    assert m.name == "unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M"
    assert m.revision == "2" * 40
    main = next(f for f in m.files if f.role == "model")
    import hashlib

    assert main.digest == hashlib.sha256(new).hexdigest()  # not the xet hash
    assert any(f.role == "projector" for f in m.files)
    kinds = sorted(lo.kind for lo in res.leftovers)
    assert kinds == ["incomplete", "old-revision"]


def test_hf_mlx_folder_model(env):
    hub = env["hf"]
    rev = "c" * 40
    add_hf_file(
        hub, "mlx-community/Qwen3-0.6B-4bit", "model.safetensors", b"\0" * 4096, revision=rev
    )
    cfg = b'{"architectures":["Qwen3ForCausalLM"],"num_hidden_layers":28,"num_attention_heads":16,"num_key_value_heads":8,"head_dim":128,"hidden_size":1024,"max_position_embeddings":40960,"quantization":{"group_size":64,"bits":4}}'
    add_hf_file(hub, "mlx-community/Qwen3-0.6B-4bit", "config.json", cfg, revision=rev)
    add_hf_file(
        hub,
        "mlx-community/Qwen3-0.6B-4bit",
        "tokenizer_config.json",
        b'{"chat_template": "{{ x }}"}',
        revision=rev,
    )
    res = huggingface.scan(hub)
    m = res.models[0]
    assert (m.format, m.template_source, m.arch.n_layers) == ("mlx", "tokenizer_config.json", 28)
    assert "tokenizer_config.json" in m.meta["files"]


def test_lmstudio_folder_with_mmproj_and_shards(env, tmp_path):
    root = env["lmstudio"]
    d = root / "lmstudio-community" / "gemma-GGUF"
    write_gguf(d / "gemma-Q4_K_M.gguf")
    write_gguf(d / "mmproj-gemma-F16.gguf", mmproj=True)
    s = root / "someone" / "big-GGUF"
    write_gguf(s / "big-00001-of-00002.gguf")
    write_gguf(s / "big-00002-of-00002.gguf", seed=2)
    (s / "other.gguf.part").write_bytes(b"1")
    res = folders.scan(root, "lmstudio")
    by = {m.name: m for m in res.models}
    assert set(by) == {"lmstudio-community/gemma-GGUF/gemma-Q4_K_M", "someone/big-GGUF/big"}
    gem = by["lmstudio-community/gemma-GGUF/gemma-Q4_K_M"]
    assert [f.role for f in gem.files] == ["model", "projector"]
    assert gem.source_repo == "lmstudio-community/gemma-GGUF"
    assert len(by["someone/big-GGUF/big"].files) == 2
    assert res.leftovers[0].kind == "incomplete"


def test_discover_skips_missing_and_dedupes(env, tmp_path):
    assert discover() == []
    env["ollama"].mkdir(parents=True)
    extra = tmp_path / "models"
    extra.mkdir()
    locs = discover([extra, extra])
    assert [(loc.name) for loc in locs] == ["ollama", "path"]
    assert time.time() > 0
