# llm-doctor

`brew doctor` for your local AI setup. Finds duplicate weights across Ollama, LM Studio, Hugging Face and MLX, stale chat templates, outdated models and orphan blobs, moves your Ollama library to llama.cpp without re-downloading, and tells you why your coding agent misbehaves on your local endpoint.

On my MacBook Air, Ollama 0.34.4 served qwen3:1.7b with a 4,096-token window, a tenth of the 40,960 tokens the model supports. A 7,000-token prompt came back HTTP 200 with only its last 2,050 tokens seen by the model. No error, no warning.[^1]

[![CI](https://github.com/Arthur031221/llm-doctor/actions/workflows/ci.yml/badge.svg)](https://github.com/Arthur031221/llm-doctor/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/llm-doctor.svg)](https://pypi.org/project/llm-doctor/)

![llm-doctor scan, then an endpoint probe against Ollama](demo/demo.gif)

## Why

- Every runtime keeps its own copy of the same GGUF. Ollama hides it behind a sha256 blob name, so you cannot even see that LM Studio holds the same 20 GB file.
- Quantizers fix chat templates after release. Your old download keeps the broken one, and nothing tells you. Tool calls then fail in ways that look like model stupidity.
- Ollama defaults to a small context window and truncates silently. The OpenAI-compatible endpoint that coding agents use cannot raise it per request.
- Leaving Ollama means re-downloading everything, because its blobs have no names.

## Install

The PyPI release is coming. For now, install straight from GitHub:

```sh
uv tool install git+https://github.com/Arthur031221/llm-doctor
```

Or run it once without installing: `uvx --from git+https://github.com/Arthur031221/llm-doctor llm-doctor`.

Once the PyPI release is up, `uvx llm-doctor` (or `pipx install llm-doctor`, or `pip install llm-doctor`) will work directly. Python 3.10 or newer.

## Quick start

```sh
llm-doctor                                   # scan every model store
llm-doctor fix                               # preview dedupe and cleanup, changes nothing
llm-doctor fix --yes                         # apply it
llm-doctor unbundle --to llama-server        # expose Ollama models to llama.cpp
llm-doctor endpoint http://localhost:11434   # agent-readiness probe
```

A scan of my machine, with Ollama's qwen3 models plus fixtures I downloaded for testing:[^2]

```
llm-doctor scan  3 stores, 7 models, 7.0 GB on disk (4.7 s)

store        root                      models  size
ollama       ~/.ollama/models          3       4.3 GB
lmstudio     ~/.lmstudio/models        1       396.7 MB
huggingface  ~/.cache/huggingface/hub  3       2.3 GB

Findings
WARN  duplicate         hf.co/unsloth/Qwen3-0.6B-GGUF:Q4_K_M, unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M
      2 copies of the same 396.7 MB file in lmstudio, ollama [396.7 MB]
      fix: llm-doctor fix --yes
WARN  outdated          unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M
      unsloth/Qwen3-0.6B-GGUF has a newer Qwen3-0.6B-Q4_K_M.gguf
      fix: hf download unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-Q4_K_M.gguf
WARN  template-stale    unsloth/Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M
      chat template differs from unsloth/Qwen3-0.6B-GGUF on the Hub (local 8428c815ac94, upstream 5da44855ab7e, +7/-6 lines)
      fix: hf download unsloth/Qwen3-0.6B-GGUF Qwen3-0.6B-Q4_K_M.gguf
```

`llm-doctor fix --yes` then replaced the LM Studio copy with a symlink to the Ollama blob and reclaimed 396.7 MB. The next scan reported no duplicates.

The stale template is a real one. The 2025-05-09 revision of that GGUF reads `message.content` without checking for null, and assistant turns that carry only tool calls often have null content. A later upload fixed it.

The endpoint probe against the same Ollama, default settings:[^1]

```
Context: 4,096 effective of 40,960 advertised, FAIL

check            result  detail
context          FAIL    4,096 effective of 40,960 advertised (server loaded 4,096). front of a 6,973-token prompt was dropped, server kept 2,050 tokens
tool call        PASS    Read(file_path=..., limit=..., offset=...) with valid arguments
tool result      PASS    tool result accepted and used
parallel tools   PASS    2 calls in one response
streaming tools  PASS    tool_calls arrived in 1 stream delta, arguments valid
json schema      PASS    output matches the schema
think tags       PASS    reasoning returned separately in `reasoning`
max_tokens       PASS    stopped at 16 tokens, finish_reason length
prefix cache     PASS    TTFT 3.29 s then 0.35 s on a repeated 2,490-token prefix, 2,475 tokens cached
speed 1k         INFO    TTFT 1.85 s, prefill 532 tok/s, decode 40.8 tok/s
speed 8k         SKIP    a prompt of 8,192 tokens does not fit the 4,096-token window
ram headroom     PASS    3.7 GiB available of 24.0 GiB, model resident 1.7 GiB

Fixes for ollama
1. Raise Ollama's context window. Ollama loaded qwen3:1.7b with a 4,096-token window. This model needs about 3.5 GiB of KV cache at 32,768 tokens. Server-wide: `OLLAMA_CONTEXT_LENGTH=32768 ollama serve`, or for the menu bar app `launchctl setenv OLLAMA_CONTEXT_LENGTH 32768` and restart Ollama.
2. Or per model: `printf 'FROM qwen3:1.7b\nPARAMETER num_ctx 32768\n' > Modelfile && ollama create qwen3-32k -f Modelfile`, then point the agent at the new name.
3. The OpenAI-compatible /v1 endpoint has no num_ctx field, so a client cannot raise the window per request. Ollama's native /api/chat accepts options.num_ctx.
```

After applying fix 2, the same probe verified 16,384 tokens with no truncation and every check passed.

## How it works

**Stores.** Ollama manifests (`manifests/<host>/<namespace>/<model>/<tag>`) list layers by media type: weights, projector, adapter, params, template, system. Blobs that no manifest references are orphans. `-partial` files are unfinished pulls. The Hugging Face cache is read from `refs`, `snapshots` and `blobs`, including the huggingface_hub 2.0 layout where each repo blob is a symlink into a shared, xet-hashed store. LM Studio folders follow `<publisher>/<repo>/<file>.gguf`. Folders with `config.json` next to `*.safetensors` are MLX or transformers models.

**Duplicates.** Files are grouped by size first. Only files that share a size with another physical file get hashed, and only if the store does not already name them by sha256. Ollama and Hugging Face LFS blobs are named by their sha256, so in practice only LM Studio and loose files are ever read. Hashes are cached in `~/.llm-doctor/cache.json` keyed by path, size, mtime and inode. Hardlinks and existing symlinks count once.

**GGUF headers.** llm-doctor parses GGUF metadata itself and skips tokenizer arrays by length. On the 1.4 GB qwen3:1.7b blob that takes 39 ms, against 2.0 s for gguf-py's `GGUFReader`.[^3] The test suite checks the parser against gguf-py on generated files.

**Memory fit.** Weights plus an f16 KV cache at the target context (`--ctx`, default 32,768, capped at the model's trained context). KV bytes are `layers x kv_heads x (key_dim + value_dim) x 2 x tokens`, with per-layer KV heads, sliding-window layers and MLA handled when the metadata describes them. The result is compared with physical RAM and with the macOS GPU wired limit (`iogpu.wired_limit_mb`, or about two thirds of RAM when unset on machines up to 36 GB).

**Templates and updates.** For Hugging Face GGUFs the Hub's `paths-info` API gives the upstream sha256 of the same file. If it differs, llm-doctor reads the upstream GGUF header with an HTTP range request (no weights downloaded) and compares chat templates after whitespace normalization. It also compares with the base model's `tokenizer_config.json` or `chat_template.jinja` when the GGUF or model card names one, and reports that as a hint only, since quantizers patch templates on purpose. For Ollama, the registry manifest for the same tag is compared layer by layer, so a template-only update is reported as a few KB to pull.

**Fix.** Dry run by default. Duplicates are replaced by a symlink (or `--hardlink`) to one kept copy. The kept copy comes from the Hugging Face cache first, then Ollama, then LM Studio and plain folders. Files inside the Hugging Face cache are never rewritten, because huggingface_hub reference-counts them. Orphans and partial downloads are deleted only when untouched for `--min-age` minutes, so a pull in progress is safe. Every change goes to `~/.llm-doctor/fix-log.jsonl`.

**Unbundle.** Each Ollama model becomes a named symlink to its blob. Vision projectors go next to the model as `mmproj-<name>.gguf` in their own folder, the layout `llama-server --models-dir` expects. Modelfile parameters map to llama-server flags (`num_ctx` to `ctx-size`, `temperature` to `temp`, `top_k`, `top_p`, `min_p`, `repeat_penalty` and a dozen more). I checked the preset INI this writes against llama-server b11146 in router mode, and the endpoint probe then measured exactly the context it set.

**Endpoint probe.** A calibration request learns the server's characters per token. The context probe then sends prompts of 85 percent of 2k, 4k, 8k and 16k tokens (32k with `--deep`), each with a random code at the very start and another at the very end, and asks for both. The server's own `prompt_tokens` count is the main signal: when it is far below what was sent, the prompt was truncated, and the missing code tells you which end was cut. A model that simply forgets a code while the server counted every token is reported as weak recall, not truncation. Tool probes use five tools shaped like a coding agent's (Read, Write, Edit, Bash, Grep) and validate every argument object against its JSON schema. Every request is capped by a shared time budget (90 s by default).

## Compared with

| Tool | What it covers | What it does not do |
|---|---|---|
| llm-doctor | Ollama, LM Studio, HF cache and MLX scan, dedupe on disk, orphans, stale templates, updates, Ollama export, endpoint probe with fixes | Download or update models itself (it prints the command), run models |
| [sammcj/gollama](https://github.com/sammcj/gollama) | Ollama model manager TUI: list, sort, delete, copy, edit Modelfiles, vRAM and context estimates | Other stores (LM Studio linking was removed in v2.0.1), duplicate or orphan detection, update or template checks, endpoint probing |
| [llamastash](https://github.com/llamastash/llamastash) | Runtime manager: finds GGUFs in HF, Ollama and LM Studio caches, KV-aware memory estimates, mmproj pairing, launches and routes models | Its dedupe collapses symlinks in its own model list rather than reclaiming disk. No upstream template or version checks, no endpoint probe |
| [ollama-to-lmstudio-symlinks](https://github.com/qaribhaider/ollama-to-lmstudio-symlinks) | Links models between Ollama and LM Studio, pairs mmproj files, groups shards, cleans broken links | Other stores, parameter translation, llama-server or llama-swap configs, template or update checks |
| [chatpin](https://github.com/cloudpayload/chatpin) | Pins a reviewed chat template, diffs against an upstream you choose, fails CI when it changes | Finding which of your models are stale on its own, scanning stores |
| `hf cache ls` / `hf cache prune` (huggingface_hub) | Sizes, revisions and cleanup inside the Hugging Face cache | Anything outside that cache, cross-store duplicates, template checks |
| [compatcanary](https://github.com/CognizenOrg/compatcanary) | OpenAI-compatibility probes: model list, chat, streaming, tool calling, structured output, Responses API, with a score | Context truncation tests, reasoning leakage, prefix cache and speed, backend-specific fixes |

## Command reference

Every command takes `--json` and `-h/--help`.

### `llm-doctor` / `llm-doctor scan`

| Option | Default | |
|---|---|---|
| `-p, --path DIR` | | Extra folder with GGUF or MLX models, repeatable |
| `--ctx N` | 32768 | Context for the memory estimate |
| `--offline` | off | Skip Ollama registry and Hugging Face Hub lookups |
| `--no-hash` | off | Skip duplicate detection |
| `--min-size MB` | 16 | Ignore smaller files when looking for duplicates |
| `-v, --verbose` | off | Also show info notes, such as template differences from the base model |

Finding codes: `duplicate`, `orphan`, `incomplete`, `old-revision`, `broken`, `no-fit`, `tight-fit`, `template-stale`, `template-missing`, `template-vs-base`, `outdated`, `newer-revision`, `upstream-missing`, `store-problem`.

### `llm-doctor fix`

| Option | Default | |
|---|---|---|
| `-y, --yes` | off | Apply. Without it nothing changes |
| `--hardlink` | off | Hardlinks instead of symlinks (same filesystem only) |
| `--only KIND` | all | `dedupe`, `orphans`, `incomplete`, repeatable or comma-separated |
| `--min-age MIN` | 60 | Minutes a leftover must be untouched before deletion |
| `-p, --path DIR`, `--min-size MB` | | As for scan |

### `llm-doctor unbundle`

| Option | Default | |
|---|---|---|
| `--to TARGET` | llama-server | `llama-server` (presets INI), `llama-swap` (config YAML), `lmstudio` (folder layout), `mlx` (conversion script) |
| `--out DIR` | `./ollama-unbundled` | For `lmstudio` the default is `~/.lmstudio/models/ollama` |
| `-m, --model NAME` | all | Only this Ollama model, repeatable |
| `--hardlink` | off | Hardlink instead of symlink, survives `ollama rm` |
| `--ctx N` | 32768 | Context to write when the Modelfile sets no `num_ctx` |
| `--llama-server PATH` | from PATH | Binary used in llama-swap commands |
| `--dry-run` | off | Show the plan, write nothing |

Then run `llama-server --models-preset ollama-unbundled/presets.ini` or `llama-swap --config ollama-unbundled/llama-swap.yaml`.

### `llm-doctor endpoint URL`

| Option | Default | |
|---|---|---|
| `-m, --model NAME` | loaded or first model | Model to test |
| `--api` | openai | `openai` for `/v1/chat/completions`, `ollama` for `/api/chat` |
| `--deep` | off | Add the 32k context level |
| `--budget S` | 90, or 300 with `--deep` | Time budget for all probes |
| `--only PROBE` | all | `context`, `tools`, `json`, `think`, `max-tokens`, `prefix-cache`, `speed`, `ram` |
| `--api-key KEY` | | Bearer token, also read from `LLM_DOCTOR_API_KEY` |

Exits with status 1 when any check fails, so it works in scripts. Backends are detected from `/api/version` (Ollama), `/props` (llama-server, including router mode) and `/api/v0/models` (LM Studio). Anything else is treated as a generic OpenAI-compatible server.

### Environment

| Variable | Used for |
|---|---|
| `OLLAMA_MODELS` | Ollama store location |
| `HF_HUB_CACHE`, `HF_HOME` | Hugging Face cache location |
| `HF_TOKEN`, `HF_ENDPOINT` | Gated repos and mirrors for upstream checks |
| `LLM_DOCTOR_PATHS` | Extra model folders, separated by `:` |
| `LLM_DOCTOR_HOME` | Where the hash cache and fix log live, default `~/.llm-doctor` |
| `LLM_DOCTOR_API_KEY` | Bearer token for `endpoint` |

## Limits and FAQ

**Does it download or update models?** No. It tells you what is stale and prints the `ollama pull` or `hf download` command.

**Is `fix` safe?** It only links files whose sha256 matches and only deletes orphans and partial downloads older than `--min-age`. The trade-off with symlinks: if you later `ollama rm` the kept copy, the linked copies break. Use `--hardlink` if your stores share a filesystem and you want each store to stay independent.

**Does it support Ollama's newer non-GGUF layouts?** Manifests with unknown layer types are scanned, sized and checked for orphans. `unbundle` only exports GGUF weights and says so for anything else.

**Why does `--to mlx` write a script instead of files?** mlx-lm cannot load Ollama's GGUF k-quants. The script runs `mlx_lm.convert` on the original Hugging Face repo when the GGUF metadata names it.

**How exact is the memory estimate?** Weights plus f16 KV cache. It ignores compute buffers (usually a few hundred MB), quantized KV caches (q8_0 halves the KV number) and runtime overhead. Treat `tight` as a warning, not a verdict.

**How exact is the template check?** It is a text comparison after whitespace normalization. It does not render templates, so two templates that differ only in formatting inside a Jinja block still count as different.

**Does the endpoint probe cost money on hosted APIs?** It sends about 17 requests and up to about 50,000 prompt tokens with default settings. Point it at paid endpoints only if you mean to.

**What does the probe not test?** Output quality, long multi-turn agent sessions, concurrency and rate limits. A small model can fail the marker test from weak recall, which is reported as a warning, not as truncation.

**Platforms.** Built and tested on macOS with Apple Silicon. The store scanners and the probe also run on Linux (CI runs there). Windows is untested.

## Related projects

- [gpuwho](https://github.com/Arthur031221/gpuwho): Shows which local process is using the GPU right now, a live view next to llm-doctor's static inventory of your model store.
- [gpuwait](https://github.com/Arthur031221/gpuwait): Measures how much of a serving window your local model spends idle, a companion number to llm-doctor's setup diagnosis.
- [ollama-verify](https://github.com/Arthur031221/ollama-verify): Checks the integrity of the Ollama blobs llm-doctor also inventories, narrower in scope and read-only.

## Contributing and license

Bug reports with `--json` output and store layouts that llm-doctor gets wrong are the most useful contributions. See [CONTRIBUTING.md](CONTRIBUTING.md). MIT licensed, see [LICENSE](LICENSE).

[^1]: Measured 2026-09-30 on a MacBook Air M5 with 24 GB, macOS 26, Ollama 0.34.4 from Homebrew with default settings, model `qwen3:1.7b` (Q4_K_M), `llm-doctor endpoint http://localhost:11434 --model qwen3:1.7b`. Three full runs found the same truncation, taking 13 s, 8 s (with `--api ollama`) and 71 s (while other builds loaded the machine). The 6,973-token figure is the probe's estimate from its calibration request. Ollama's own `prompt_tokens` for the 2k and 4k levels were within 1 percent of the estimate, and 2,050 for the 8k level. `/api/ps` reported `context_length: 4096`, and `ps` showed Ollama's runner started with `-c 4096 -np 1 --context-shift --keep 4`. With `PARAMETER num_ctx 32768` in a Modelfile, the probe verified 16,384 tokens in 57 s.
[^2]: Same machine and date. Besides the two Ollama library models, I downloaded fixtures for this test: `unsloth/Qwen3-0.6B-GGUF` Q4_K_M at revision 649a90f (May 2025) into the Hugging Face cache, `mlx-community/Qwen3-0.6B-4bit`, and `ollama pull hf.co/unsloth/Qwen3-0.6B-GGUF:Q4_K_M`, then copied that GGUF into `~/.lmstudio/models` as LM Studio would store it. The cache also held an MLX Whisper model from another project. The first scan, with an empty hash cache, took 11.8 s and hashed 396.7 MB. Later scans took 4.7 s, most of it Hub lookups.
[^3]: One run each on the same machine, `sha256-3d0b79...` (qwen3:1.7b, 1.36 GB, 27 metadata keys, 151,936-token vocabulary). gguf-py 0.19.0 also needs 0.6 s to import.
