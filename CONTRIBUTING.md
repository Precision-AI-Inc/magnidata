# Contributing

Thank you for your interest in contributing to this Precision AI project.

## Setup

Requires Python 3.10+.

```bash
git clone git@github.com:Precision-AI-Inc/magnidata.git
cd magnidata

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -e ".[dev]"
pre-commit install
```

`pip install -e ".[dev]"` installs the API + feature-extraction toolkit plus test tooling and pre-commit. `pre-commit install` registers the hooks so they run automatically on every commit.

The dashboard is a separate Node/Vite app under `precisionai/magnidata/dashboard/` — see its own setup in the [README](README.md#dashboard). Running the full stack (API + dashboard) is done via `docker compose up --build` from the repo root.

Offline embeddings generation and learned image-quality scoring (`tools/embeddings.py`, `tools/nima.py`) need heavier ML dependencies not required by the API itself — install them with `pip install -e ".[ml]"` when you need those CLI tools.

## Making changes

1. Create a branch from `main`.
2. Make your changes.
3. Add or update tests for anything you touch.
4. Commit. Pre-commit hooks run automatically and will block the commit if any check fails.

If a hook auto-fixes files (ruff lint/format), stage the changes and commit again.

## Pre-commit hooks

| Hook | What it checks |
|---|---|
| File hygiene | Large files (>1800 KB), trailing whitespace, merge conflicts, private keys, debug statements, BOM removal |
| `ruff` | Linting and import sorting (auto-fix) on the files you touch — includes `ANN` rules that enforce PEP 484 annotations |
| `ruff-format` | Code formatting (auto-fix) |
| `pyright` | Static type checking — config in `[tool.pyright]` in `pyproject.toml` |
| `pytest` | Full test suite with a coverage floor (see **Known gaps** below) |

Run all hooks manually without committing:

```bash
pre-commit run --all-files
```

## Code style

- **Formatter / linter:** ruff (`line-length = 120`, Python 3.10 target).
- **Type hints:** all *new* functions (public and private) should have complete PEP 484 type annotations. `Any` is allowed for genuinely dynamic types.
- **Docstrings:** NumPy style for all *new* public functions, classes, and modules.
- **Comments:** only where the _why_ is non-obvious. No inline narration of what the code does.

See [CLAUDE.md](CLAUDE.md) for the full standard this project targets.

## Known gaps

This codebase was migrated from an existing, working application rather than started from this template, so it does not yet meet every standard in `CLAUDE.md`. These are tracked, visible gaps rather than silently-ignored ones:

- **Test coverage is ~66%, not the org's 90% target.** The `pytest` pre-commit hook enforces 60% (`--cov-fail-under` in `pyproject.toml`) so the hook reflects reality instead of being disabled outright. Raising this is ongoing work — please add coverage for whatever you touch rather than letting it slip further.
- **`ruff check --all-files` and `pyright` are both clean** as of the docstring/type-hint/complexity cleanup that closed out the previous backlog here. A handful of bandit (`S`) findings are pre-ignored with inline justification rather than fixed in code, because the underlying rules do no control-flow analysis (`S310` on `urlopen()`, `S603`/`PLC0415` in `prepare_coco128_dashboard.py`) — see the comments in `pyproject.toml`'s `[tool.ruff.lint]` and `per-file-ignores` sections. If either check starts reporting a backlog again, please don't let it grow silently — fix as you go, the way `S311`/`S104` are handled.
- **Flask, not FastAPI.** `CLAUDE.md`'s example layout assumes FastAPI; this project's API (`precisionai/magnidata/api/`) is Flask and is not laid out in the `api/routes/ + schemas/ + services/ + metrics/` shape the template describes. A full port is out of scope for the migration that produced this repo.
- **Tests are colocated with source** (`precisionai/magnidata/**/test_*.py`), not under a top-level `tests/` mirroring the package — `pyproject.toml`'s `testpaths` and the ruff `per-file-ignores` are configured for this layout rather than the template's default.

## Tests

```bash
pytest                                          # run all tests with coverage report
pytest precisionai/magnidata/tools/test_features.py  # run a specific file
pytest -k "coverage"                            # run tests matching a pattern
```

Tests live next to the module they cover (see **Known gaps** above). The coverage floor is enforced both by `pytest` directly and by the pre-commit hook.

## Pull requests

- Keep PRs focused — one logical change per PR.
- Write a clear description of what changed and why.
- All pre-commit hooks must pass before requesting review.

## License

By contributing, you agree that your contributions are licensed under the Apache License, Version 2.0.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
