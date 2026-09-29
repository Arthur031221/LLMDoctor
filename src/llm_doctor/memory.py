"""KV cache and memory fit estimates from GGUF metadata or config.json."""

from __future__ import annotations

from dataclasses import dataclass

from llm_doctor.gguf_meta import ArrayInfo, GGUFHeader, scalar

F16 = 2


@dataclass
class ArchInfo:
    arch: str
    n_layers: int
    kv_heads: list[int]  # per layer, 0 for layers without attention (recurrent or SSM)
    head_dim_k: int
    head_dim_v: int
    ctx_train: int | None = None
    sliding_window: int | None = None
    swa_layers: list[bool] | None = None
    mla_dim: int | None = None  # elements per token per layer for MLA models

    def kv_bytes(self, ctx: int, bytes_per_elem: int = F16) -> int:
        total = 0
        for i in range(self.n_layers):
            layer_ctx = ctx
            swa = self.swa_layers
            if self.sliding_window and swa and i < len(swa) and swa[i]:
                layer_ctx = min(ctx, self.sliding_window)
            if self.mla_dim:
                total += self.mla_dim * layer_ctx * bytes_per_elem
                continue
            heads = self.kv_heads[i] if i < len(self.kv_heads) else self.kv_heads[-1]
            total += heads * (self.head_dim_k + self.head_dim_v) * layer_ctx * bytes_per_elem
        return total

    def to_dict(self) -> dict:
        return {
            "arch": self.arch,
            "layers": self.n_layers,
            "kv_heads": max(self.kv_heads) if self.kv_heads else 0,
            "head_dim": self.head_dim_k,
            "context_train": self.ctx_train,
            "sliding_window": self.sliding_window,
        }


def _per_layer(value, n_layers: int, default: int) -> list[int]:
    if isinstance(value, list) and value:
        vals = [int(v) for v in value]
        if len(vals) < n_layers:
            vals += [vals[-1]] * (n_layers - len(vals))
        return vals[:n_layers]
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [int(value)] * n_layers
    return [default] * n_layers


def _swa_pattern(arch: str, n_layers: int, pattern) -> list[bool] | None:
    if isinstance(pattern, list) and pattern and all(isinstance(p, bool) for p in pattern):
        return (pattern + [pattern[-1]] * n_layers)[:n_layers]
    if isinstance(pattern, int) and pattern > 1:
        return [(i + 1) % pattern != 0 for i in range(n_layers)]
    # llama.cpp hardcodes these for older conversions without a pattern key.
    if arch.startswith("gemma3"):
        return [(i + 1) % 6 != 0 for i in range(n_layers)]
    if arch == "gemma2":
        return [i % 2 == 0 for i in range(n_layers)]
    return None


def arch_from_gguf(h: GGUFHeader) -> ArchInfo | None:
    arch = h.arch
    if not arch or h.is_projector:
        return None
    n_layers = scalar(h.arch_key("block_count"))
    n_head_raw = h.arch_key("attention.head_count")
    n_embd = scalar(h.arch_key("embedding_length"))
    if not n_layers:
        return None
    n_layers = int(n_layers)
    n_head = scalar(n_head_raw, 0) or 0
    kv_raw = h.arch_key("attention.head_count_kv", n_head_raw)
    if isinstance(kv_raw, ArrayInfo):
        kv_raw = n_head
    kv_heads = _per_layer(kv_raw, n_layers, int(n_head))
    k_len = h.arch_key("attention.key_length")
    v_len = h.arch_key("attention.value_length")
    if not k_len:
        k_len = int(n_embd // n_head) if n_embd and n_head else 0
    if not v_len:
        v_len = k_len
    mla_dim = None
    kv_lora = h.arch_key("attention.kv_lora_rank")
    if kv_lora:
        rope = h.arch_key("rope.dimension_count") or 64
        mla_dim = int(kv_lora) + int(rope)
    swa = h.arch_key("attention.sliding_window")
    swa = int(scalar(swa)) if swa else None
    swa_layers = _swa_pattern(arch, n_layers, h.arch_key("attention.sliding_window_pattern"))
    ctx = h.arch_key("context_length")
    return ArchInfo(
        arch=arch,
        n_layers=n_layers,
        kv_heads=kv_heads,
        head_dim_k=int(k_len or 0),
        head_dim_v=int(v_len or 0),
        ctx_train=int(ctx) if ctx else None,
        sliding_window=swa if swa_layers else None,
        swa_layers=swa_layers if swa else None,
        mla_dim=mla_dim,
    )


def arch_from_config(cfg: dict) -> ArchInfo | None:
    """Build ArchInfo from a Hugging Face or MLX config.json."""
    arch = (cfg.get("architectures") or [cfg.get("model_type") or "unknown"])[0]
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else None
    c = {**cfg, **text} if text else cfg
    n_layers = c.get("num_hidden_layers") or c.get("n_layer") or c.get("num_layers")
    n_head = c.get("num_attention_heads") or c.get("n_head")
    if not n_layers or not n_head:
        return None
    n_layers, n_head = int(n_layers), int(n_head)
    n_kv = int(c.get("num_key_value_heads") or n_head)
    hidden = c.get("hidden_size") or c.get("n_embd") or 0
    head_dim = int(c.get("head_dim") or (hidden // n_head if hidden else 0))
    kv_heads = [n_kv] * n_layers
    layer_types = c.get("layer_types")
    swa_layers = None
    if isinstance(layer_types, list) and len(layer_types) >= n_layers:
        swa_layers = [t == "sliding_attention" for t in layer_types[:n_layers]]
        kv_heads = [
            0 if t not in ("full_attention", "sliding_attention", "attention") else n_kv
            for t in layer_types[:n_layers]
        ]
        if not any(kv_heads):
            kv_heads = [n_kv] * n_layers
    swa = c.get("sliding_window") if c.get("use_sliding_window", True) else None
    mla_dim = None
    if c.get("kv_lora_rank"):
        mla_dim = int(c["kv_lora_rank"]) + int(c.get("qk_rope_head_dim") or 64)
    if swa and swa_layers is None:
        swa_layers = _swa_pattern(
            str(c.get("model_type", "")), n_layers, c.get("sliding_window_pattern")
        )
    ctx = c.get("max_position_embeddings") or c.get("max_sequence_length")
    return ArchInfo(
        arch=str(arch),
        n_layers=n_layers,
        kv_heads=kv_heads,
        head_dim_k=head_dim,
        head_dim_v=head_dim,
        ctx_train=int(ctx) if ctx else None,
        sliding_window=int(swa) if swa and swa_layers else None,
        swa_layers=swa_layers if swa else None,
        mla_dim=mla_dim,
    )


@dataclass
class FitEstimate:
    context: int
    weights: int
    kv: int
    total: int
    budget: int | None
    ram: int | None
    status: str  # fits, tight, no-fit, unknown

    def to_dict(self) -> dict:
        return {
            "context": self.context,
            "weights_bytes": self.weights,
            "kv_bytes": self.kv,
            "total_bytes": self.total,
            "gpu_budget_bytes": self.budget,
            "ram_bytes": self.ram,
            "status": self.status,
        }


def estimate_fit(
    arch: ArchInfo | None, weights: int, ctx: int, budget: int | None, ram: int | None
) -> FitEstimate:
    if arch is None:
        return FitEstimate(ctx, weights, 0, weights, budget, ram, "unknown")
    eff_ctx = min(ctx, arch.ctx_train) if arch.ctx_train else ctx
    kv = arch.kv_bytes(eff_ctx)
    total = weights + kv
    if ram and total > ram:
        status = "no-fit"
    elif budget and total > budget:
        status = "tight"
    elif budget or ram:
        status = "fits"
    else:
        status = "unknown"
    return FitEstimate(eff_ctx, weights, kv, total, budget, ram, status)
