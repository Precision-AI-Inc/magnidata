# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Guard the order of steps and the files shipped in the Hugging Face Space Dockerfile.

The image build loads the datasets through the same endpoint demo mode refuses, so demo mode may only
be switched on after that step; set earlier, the build fails with "Disabled for the Demo". The API
serves GET /api/openapi.json from the openapi.yaml beside app.py, so the image must copy it there.

Run:  python -m pytest precisionai/magnidata/api/test_space_dockerfile.py -q
"""

import posixpath
import shlex
from fnmatch import fnmatch
from pathlib import Path

import pytest

DOCKERFILE = Path(__file__).resolve().parents[3] / "deploy" / "huggingface" / "Dockerfile"


@pytest.fixture
def lines():
    if not DOCKERFILE.is_file():
        pytest.skip("deploy/huggingface/Dockerfile is not part of this checkout")
    return DOCKERFILE.read_text().splitlines()


def _first(lines, needle):
    return next(i for i, line in enumerate(lines) if needle in line and not line.lstrip().startswith("#"))


def test_demo_mode_is_switched_on_only_after_the_datasets_are_loaded(lines):
    assert _first(lines, "MAGNIDATA_DEMO_MODE=1") > _first(lines, "python preload_demo.py")


def test_demo_mode_is_set_exactly_once(lines):
    assert sum("MAGNIDATA_DEMO_MODE" in line and not line.lstrip().startswith("#") for line in lines) == 1


def test_the_space_is_served_by_gunicorn_not_the_development_server(lines):
    start = _first(lines, "CMD [")
    command = " ".join(line.strip() for line in lines[start:]).replace("\\", " ")
    assert '"gunicorn"' in command
    assert '"app:app"' in command
    assert "app.py" not in command
    assert any("pip install" in line and "gunicorn==" in line for line in lines)


def test_gunicorn_runs_a_single_worker_because_analysis_jobs_live_in_process_memory(lines):
    start = _first(lines, "CMD [")
    command = " ".join(line.strip() for line in lines[start:])
    assert '"--workers", "1"' in command


def _instructions(lines):
    """Return the Dockerfile's instructions as (keyword, arguments), with continuation lines joined."""
    instructions, pending = [], ""
    for line in lines:
        if not pending and (not line.strip() or line.lstrip().startswith("#")):
            continue
        pending += line.strip()
        if pending.endswith("\\"):
            pending = pending[:-1] + " "
            continue
        keyword, _, arguments = pending.partition(" ")
        instructions.append((keyword.upper(), arguments.strip()))
        pending = ""
    return instructions


def _final_stage_copies(lines):
    """Return each COPY of the final stage as (sources, absolute destination), resolved against its WORKDIR."""
    instructions = _instructions(lines)
    start = max(i for i, (keyword, _) in enumerate(instructions) if keyword == "FROM")
    workdir, copies = "/", []
    for keyword, arguments in instructions[start + 1 :]:
        if keyword == "WORKDIR":
            workdir = posixpath.join(workdir, arguments)
        elif keyword == "COPY":
            *sources, destination = [arg for arg in shlex.split(arguments) if not arg.startswith("--")]
            copies.append((sources, posixpath.join(workdir, destination)))
    return copies


def _landing_directory(copies, source):
    """Return the directory the final stage copies the cloned file ``source`` into, or None when it is not copied."""
    landed = None
    for sources, destination in copies:
        if any(fnmatch(source, pattern) for pattern in sources):
            renamed = sources == [source] and not destination.endswith("/")
            landed = posixpath.dirname(destination) if renamed else posixpath.normpath(destination)
    return landed


def test_the_openapi_document_lands_next_to_app_py(lines):
    # app.py reads OPENAPI_PATH from its own directory, and COPY api/*.py does not match openapi.yaml.
    copies = _final_stage_copies(lines)
    api_directory = _landing_directory(copies, "/src/precisionai/magnidata/api/app.py")
    assert api_directory == "/app"
    assert _landing_directory(copies, "/src/precisionai/magnidata/api/openapi.yaml") == api_directory
