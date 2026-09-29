from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


@pytest.fixture
def env(tmp_path, monkeypatch):
    """Point every store at an empty temporary home."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("LLM_DOCTOR_HOME", str(home / ".llm-doctor"))
    monkeypatch.setenv("OLLAMA_MODELS", str(home / ".ollama" / "models"))
    monkeypatch.setenv("HF_HUB_CACHE", str(home / ".cache" / "huggingface" / "hub"))
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("LLM_DOCTOR_PATHS", raising=False)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    return {
        "home": home,
        "ollama": home / ".ollama" / "models",
        "hf": home / ".cache" / "huggingface" / "hub",
        "lmstudio": home / ".lmstudio" / "models",
        "tmp": tmp_path,
    }
