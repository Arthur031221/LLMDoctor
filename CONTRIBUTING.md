# Contributing

Bug reports with `--json` output are the most useful contribution. Store layouts drift, so a
directory listing of a layout llm-doctor gets wrong is worth a lot.

## Setup

```sh
git clone https://github.com/Arthur031221/llm-doctor
cd llm-doctor
uv sync
uv run llm-doctor --help
```

## Before you open a pull request

```sh
uv run ruff check .
uv run ruff format .
uv run pytest
```

The test suite builds tiny synthetic stores and a fake chat server, so it needs no models and
no network. It runs in well under a minute. Keep it that way: new tests should use the builders
in `tests/helpers.py` and `tests/fake_server.py`.

## Guidelines

- `fix` must never touch a file it cannot prove is a duplicate or a leftover. When in doubt, skip
  and say why.
- Every new check needs a `--json` field and a test.
- Keep dependencies to what is in `pyproject.toml` unless there is no reasonable alternative.
- Commit messages: imperative mood, one logical change per commit.
