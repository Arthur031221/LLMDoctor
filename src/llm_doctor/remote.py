"""Network lookups: Hugging Face Hub API and Ollama-compatible registries.

Every call is short, memoized, and failures turn into None plus a note in `errors`,
so an offline machine or a rate limit never breaks a scan.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path
from urllib.parse import quote

import httpx

from llm_doctor import __version__
from llm_doctor.gguf_meta import GGUFError, GGUFHeader, Truncated, parse_buffer

MANIFEST_ACCEPT = "application/vnd.docker.distribution.manifest.v2+json"
HEADER_RANGES = (8 << 20, 32 << 20)


def hf_token() -> str | None:
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(var):
            return os.environ[var].strip()
    home = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")).expanduser()
    try:
        tok = (home / "token").read_text().strip()
        return tok or None
    except OSError:
        return None


class Remote:
    def __init__(self, timeout: float = 8.0, transport: httpx.BaseTransport | None = None) -> None:
        self.hf = os.environ.get("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
        self.client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": f"llm-doctor/{__version__}"},
            transport=transport,
        )
        self._token = hf_token()
        self._memo: dict = {}
        self._lock = threading.Lock()
        self.errors: list[str] = []

    def close(self) -> None:
        self.client.close()

    def _note(self, msg: str) -> None:
        with self._lock:
            if msg not in self.errors:
                self.errors.append(msg)

    def _hf_headers(self, url: str) -> dict:
        if self._token and url.startswith(self.hf):
            return {"Authorization": f"Bearer {self._token}"}
        return {}

    def _get(self, url: str, **kw) -> httpx.Response | None:
        headers = {**self._hf_headers(url), **kw.pop("headers", {})}
        try:
            r = self.client.get(url, headers=headers, **kw)
        except httpx.HTTPError as e:
            self._note(f"{url.split('?')[0]}: {type(e).__name__}")
            return None
        return r

    def _memo_get(self, key, fn):
        with self._lock:
            if key in self._memo:
                return self._memo[key]
        val = fn()
        with self._lock:
            self._memo[key] = val
        return val

    # Hugging Face

    def hf_model_info(self, repo: str) -> dict | None:
        def fetch():
            url = f"{self.hf}/api/models/{repo}"
            r = self._get(
                url, params=[("expand[]", "sha"), ("expand[]", "gguf"), ("expand[]", "cardData")]
            )
            if r is None or r.status_code != 200:
                if r is not None and r.status_code not in (401, 404):
                    self._note(f"HF model info {repo}: HTTP {r.status_code}")
                return None
            try:
                return r.json()
            except ValueError:
                return None

        return self._memo_get(("info", repo), fetch)

    def hf_paths_info(self, repo: str, rev: str, paths: list[str]) -> dict[str, dict]:
        def fetch():
            url = f"{self.hf}/api/models/{repo}/paths-info/{quote(rev, safe='')}"
            try:
                r = self.client.post(
                    url, data={"paths": list(paths)}, headers=self._hf_headers(url)
                )
            except httpx.HTTPError as e:
                self._note(f"HF paths-info {repo}: {type(e).__name__}")
                return {}
            if r.status_code != 200:
                return {}
            try:
                return {e["path"]: e for e in r.json() if isinstance(e, dict) and "path" in e}
            except (ValueError, TypeError):
                return {}

        return self._memo_get(("paths", repo, rev, tuple(sorted(paths))), fetch)

    def hf_text(self, repo: str, filename: str, rev: str = "main") -> str | None:
        def fetch():
            url = f"{self.hf}/{repo}/resolve/{quote(rev, safe='')}/{quote(filename)}"
            r = self._get(url)
            if r is None or r.status_code != 200:
                return None
            return r.text

        return self._memo_get(("text", repo, filename, rev), fetch)

    def hf_template(self, repo: str, rev: str = "main") -> tuple[str | None, str | None]:
        """Chat template of a transformers-style repo: chat_template.jinja wins, as in transformers."""
        text = self.hf_text(repo, "chat_template.jinja", rev)
        if text:
            return text, "chat_template.jinja"
        raw = self.hf_text(repo, "tokenizer_config.json", rev)
        if raw:
            try:
                t = json.loads(raw).get("chat_template")
            except ValueError:
                t = None
            if isinstance(t, list):
                t = next((x.get("template") for x in t if x.get("name") == "default"), None)
            if isinstance(t, str):
                return t, "tokenizer_config.json"
        return None, None

    def hf_gguf_header(self, repo: str, filename: str, rev: str = "main") -> GGUFHeader | None:
        """Read a remote GGUF header with HTTP range requests, without downloading weights."""

        def fetch():
            url = f"{self.hf}/{repo}/resolve/{quote(rev, safe='')}/{quote(filename)}"
            for size in HEADER_RANGES:
                r = self._get(url, headers={"Range": f"bytes=0-{size - 1}"})
                if r is None or r.status_code not in (200, 206):
                    return None
                try:
                    return parse_buffer(r.content)
                except Truncated:
                    continue
                except GGUFError:
                    return None
            self._note(
                f"GGUF header of {repo}/{filename} is larger than {HEADER_RANGES[-1] >> 20} MB"
            )
            return None

        return self._memo_get(("gguf", repo, filename, rev), fetch)

    # Ollama registries

    def ollama_manifest(self, host: str, namespace: str, model: str, tag: str):
        """Return (sha256 of manifest bytes, manifest dict) or None."""

        def fetch():
            url = f"https://{host}/v2/{namespace}/{model}/manifests/{quote(tag, safe='')}"
            r = self._get(url, headers={"Accept": MANIFEST_ACCEPT})
            if r is None:
                return None
            if r.status_code != 200:
                if r.status_code not in (401, 404):
                    self._note(f"registry {host}/{namespace}/{model}:{tag}: HTTP {r.status_code}")
                return None
            try:
                data = r.json()
            except ValueError:
                return None
            return hashlib.sha256(r.content).hexdigest(), data

        return self._memo_get(("ollama", host, namespace, model, tag), fetch)
