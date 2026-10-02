# Changelog

## Unreleased

- Fix links created from relative model store paths so they resolve correctly.

## 0.1.1 (2026-09-30)

- Release workflow: removed the PyPI publish step, which needs a trusted publisher that is not
  set up yet. The workflow now only builds the sdist and wheel and attaches them to the GitHub
  release. A PyPI release is coming once the account is set up.

## 0.1.0 (2026-09-30)

First release.

- `scan`: discovers Ollama, LM Studio, the Hugging Face hub cache (including the huggingface_hub 2.0
  xet layout), MLX folders and any `--path`. Reports disk per store, byte-identical weights across
  stores, orphan Ollama blobs, partial downloads, blobs from superseded Hugging Face revisions,
  memory fit with a KV cache estimate at a target context, stale chat templates and outdated models
  (Ollama registry manifests, Hugging Face revisions).
- `fix`: dry-run by default. Replaces duplicate copies with symlinks or hardlinks, deletes orphan
  blobs and abandoned partial downloads, logs every change to `~/.llm-doctor/fix-log.jsonl`.
- `unbundle`: exposes Ollama models as named GGUF files with paired mmproj projectors and writes a
  llama-server `--models-preset` INI, a llama-swap config, an LM Studio folder, or an mlx-lm
  conversion script. Translates Modelfile parameters and warns about Ollama engine-format models
  (vision weights inside the main GGUF, built-in renderers) that llama.cpp may not load.
- `endpoint`: agent-readiness probe for OpenAI-compatible servers and Ollama's native API. Checks
  effective context with marker recall, tool calls (names, argument schemas, parallel calls,
  streaming deltas, tool results), JSON schema output, reasoning leakage, max_tokens, prefix
  caching, speed and memory headroom, then prints fixes for Ollama, llama-server and LM Studio.
- `--json` on every command.
