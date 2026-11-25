# Contributing

Thank you for your interest in contributing to this Precision AI project.

## Setup

Requires Python 3.10+.

```bash
git clone git@github.com:Precision-AI-Inc/magnidata.git
cd magnidata

python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate

pip install -e ".[dev]" -c constraints.txt
pre-commit install
```

`pip install -e ".[dev]"` installs the API + feature-extraction toolkit plus test tooling and pre-commit. `pre-commit install` registers the hooks so they run automatically on every commit. `-c constraints.txt` installs the Pillow, ruff, pyright and MLflow versions CI uses; `pip install -r requirements-dev.txt` adds the pinned ML stack from `requirements.txt` on top.

`constraints.txt` holds the exact versions that several files must agree on: torch, torchvision, timm and pyiqa (the extras, `requirements.txt` and the trainer image), MLflow (the extras, `precisionai/magnidata/api/requirements.txt` and the server image in `docker-compose.yml`), and ruff (the `dev` extra and the pre-commit hook). To change one, edit `constraints.txt` and the copies `precisionai/magnidata/test_dependency_pins.py` names when it fails.

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
- **Torch-dependent tests are skipped in CI.** The trainer, loss, data and worker end-to-end tests (including `api/test_e2e_smoke.py`) need torch, which CI does not install (`pip install -e ".[dev]" -c constraints.txt`). They skip cleanly when torch is missing and run when you install `.[training]` locally — please run them before changing anything under `precisionai/magnidata/training/`.
- **Flask, not FastAPI.** `CLAUDE.md`'s example layout assumes FastAPI; this project's API (`precisionai/magnidata/api/`) is Flask and is not laid out in the `api/routes/ + schemas/ + services/ + metrics/` shape the template describes. A full port is out of scope for the migration that produced this repo.
- **Tests are colocated with source** (`precisionai/magnidata/**/test_*.py`), not under a top-level `tests/` mirroring the package — `pyproject.toml`'s `testpaths` and the ruff `per-file-ignores` are configured for this layout rather than the template's default.
- **Not yet published to PyPI.** `CLAUDE.md`'s release flow publishes every `vX.Y.Z` tag to PyPI. Here the PyPI job in `.github/workflows/release.yml` stays off until a maintainer does the one-time setup in [Publishing to PyPI](#publishing-to-pypi); until then a release stops at a GitHub Release with the built sdist and wheel attached. Install from a source checkout (`pip install -e .`) or from a release's wheel.

## Tests

```bash
pytest                                          # run all tests with coverage report
pytest precisionai/magnidata/tools/test_features.py  # run a specific file
pytest -k "coverage"                            # run tests matching a pattern
```

Tests live next to the module they cover (see **Known gaps** above). The coverage floor is enforced both by `pytest` directly and by the pre-commit hook.

## Publishing to PyPI

A release is a tag push (`git tag vX.Y.Z && git push origin vX.Y.Z`). `.github/workflows/release.yml` runs the
tests, builds the sdist and wheel, checks that the built version matches the tag, and publishes the GitHub Release.
Once PyPI publishing is switched on, it then uploads the same two files to PyPI with
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/): PyPI accepts a short-lived token that GitHub issues
to that one job, so no PyPI password or API token is stored anywhere.

The package page on PyPI shows `README.pypi.md`, not `README.md`. Keep it to what `pip install magnidata` gives and
use absolute links in it, since PyPI cannot resolve repository-relative ones. The `package` CI job checks that it
renders (`twine check --strict`) and that the console scripts run from a fresh install of the wheel.

One-time setup, before the first upload. It needs a PyPI account with two-factor authentication that Precision AI
controls, and admin access to this repository.

1. On PyPI, signed in as the account that will own `magnidata`, open **Account settings → Publishing** and add a
   pending GitHub publisher: PyPI project name `magnidata`, owner `Precision-AI-Inc`, repository `magnidata`,
   workflow `release.yml`, environment `pypi`. The project is created by the first upload; it can be moved to a PyPI
   organization later from its settings.
2. On GitHub, under **Settings → Secrets and variables → Actions → Variables**, add the repository variable
   `PYPI_PUBLISH` with the value `true`. Optionally, under **Settings → Environments → pypi**, add required
   reviewers and limit deployments to `v*` tags.

After that, every `vX.Y.Z` tag also publishes to PyPI. Until the variable is set, the PyPI job is skipped and a
release ends at the GitHub Release, as before. A version uploaded to PyPI cannot be uploaded again, even after it is
deleted there, so a broken release is fixed with a new tag.

## Pull requests

- Keep PRs focused — one logical change per PR.
- Write a clear description of what changed and why.
- All pre-commit hooks must pass before requesting review.

## License

By contributing, you agree that your contributions are licensed under the Apache License, Version 2.0.

## Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md).
