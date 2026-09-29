from __future__ import annotations

import gguf
import pytest
from helpers import TEMPLATE_OLD, write_gguf

from llm_doctor.gguf_meta import ArrayInfo, GGUFError, Truncated, parse_buffer, read_header
from llm_doctor.memory import arch_from_config, arch_from_gguf, estimate_fit
from llm_doctor.util import human_bytes, human_mem, normalize_template, template_hash


def test_reader_matches_reference_implementation(tmp_path):
    p = write_gguf(tmp_path / "m.gguf", vocab=3000, base_repo="Qwen/Qwen3-0.6B")
    mine = read_header(p)
    ref = gguf.GGUFReader(str(p))
    for key, field in ref.fields.items():
        if key.startswith("GGUF."):
            continue
        ours = mine.metadata[key]
        if isinstance(ours, ArrayInfo):
            assert ours.count == len(field.data)
        else:
            assert ours == field.contents(), key
    assert mine.chat_template == TEMPLATE_OLD
    assert mine.arch == "qwen3"
    assert mine.tensor_count == 1


def test_large_arrays_are_skipped(tmp_path):
    h = read_header(write_gguf(tmp_path / "m.gguf", vocab=5000))
    tokens = h.metadata["tokenizer.ggml.tokens"]
    assert isinstance(tokens, ArrayInfo) and tokens.count == 5000


def test_truncated_buffer_and_bad_magic(tmp_path):
    raw = write_gguf(tmp_path / "m.gguf").read_bytes()
    with pytest.raises(Truncated):
        parse_buffer(raw[:200])
    h = parse_buffer(raw)
    assert h.header_bytes < len(raw)
    bad = tmp_path / "bad.gguf"
    bad.write_bytes(b"NOTGGUF" + raw)
    with pytest.raises(GGUFError):
        read_header(bad)
    cut = tmp_path / "cut.gguf"
    cut.write_bytes(raw[:300])
    with pytest.raises(GGUFError, match="incomplete"):
        read_header(cut)


def test_kv_cache_math_matches_qwen3():
    h = read_header_from(layers=28, kv_heads=8, head_dim=128)
    arch = arch_from_gguf(h)
    # 28 layers * 8 kv heads * (128 + 128) * 2 bytes = 114,688 bytes per token
    assert arch.kv_bytes(1) == 114_688
    assert arch.kv_bytes(32768) == 114_688 * 32768


def read_header_from(**kw):
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        return read_header(write_gguf(Path(d) / "m.gguf", **kw))


def test_config_json_arch_with_sliding_window():
    cfg = {
        "architectures": ["Gemma3ForCausalLM"],
        "model_type": "gemma3_text",
        "num_hidden_layers": 6,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "head_dim": 256,
        "hidden_size": 2048,
        "sliding_window": 1024,
        "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
        "max_position_embeddings": 131072,
    }
    arch = arch_from_config(cfg)
    per_tok = 4 * 512 * 2
    assert arch.kv_bytes(8192) == 5 * per_tok * 1024 + 1 * per_tok * 8192
    nested = arch_from_config(
        {
            "architectures": ["X"],
            "text_config": {k: v for k, v in cfg.items() if k != "architectures"},
        }
    )
    assert nested.n_layers == 6


def test_fit_status():
    h = read_header_from(layers=28, kv_heads=8, head_dim=128, ctx=40960)
    arch = arch_from_gguf(h)
    gib = 1024**3
    assert estimate_fit(arch, 1 * gib, 32768, 16 * gib, 24 * gib).status == "fits"
    assert estimate_fit(arch, 14 * gib, 32768, 16 * gib, 24 * gib).status == "tight"
    assert estimate_fit(arch, 30 * gib, 32768, 16 * gib, 24 * gib).status == "no-fit"
    # context is capped at what the model was trained for
    assert estimate_fit(arch, gib, 1_000_000, 16 * gib, 24 * gib).context == 40960
    assert estimate_fit(None, gib, 4096, None, None).status == "unknown"


def test_small_utils():
    assert human_bytes(1_500_000_000) == "1.5 GB"
    assert human_mem(3 * 1024**3) == "3.0 GiB"
    assert template_hash("a  \nb\n") == template_hash("a\nb")
    assert normalize_template("x \r\ny ") == "x\ny"
    assert template_hash(None) is None
