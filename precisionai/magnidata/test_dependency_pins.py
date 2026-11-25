# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Keep every copy of a pinned dependency version in step with ``constraints.txt``.

``constraints.txt`` at the repository root holds the exact versions that more than one install path needs: the
trainer image, the ``pyproject.toml`` extras, ``requirements.txt``, the API's own requirements file, CI, the MLflow
server image in ``docker-compose.yml`` and the ruff pre-commit hook. Files that pip reads at install time use it as a
constraints file. The others cannot read it, so these tests fail when one of them names a different version, as
happened when the trainer image installed one torch build and the ``training`` extra replaced it with another.

Run:  python -m pytest precisionai/magnidata/test_dependency_pins.py -q
"""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CONSTRAINTS = REPO / "constraints.txt"

PIN = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?==([^\s;#]+)")
QUOTED_PIN = re.compile(r"\"([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?==([^\";\s]+)\"")
# A requirement line that is a bare name (optionally with extras), so its version comes from constraints.txt.
BARE = re.compile(r"^([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?\s*(?:#.*)?$")


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _read(relative: str) -> str:
    path = REPO / relative
    if not path.is_file():
        pytest.skip(f"{relative} is not part of this checkout")
    return path.read_text()


def _requirement_lines(text: str) -> list[str]:
    lines = (line.split(" #", 1)[0].strip() for line in text.splitlines())
    return [line for line in lines if line and not line.startswith("#")]


@pytest.fixture(scope="module")
def constraints() -> dict[str, str]:
    pins = {}
    for line in _requirement_lines(_read("constraints.txt")):
        match = PIN.match(line)
        assert match, f"constraints.txt line is not an exact pin: {line!r}"
        pins[_norm(match.group(1))] = match.group(2)
    return pins


def _assert_agrees(pins: list[tuple[str, str]], constraints: dict[str, str], where: str) -> int:
    checked = 0
    for name, version in pins:
        if _norm(name) in constraints:
            assert version == constraints[_norm(name)], (
                f"{where} pins {name}=={version}, constraints.txt has {constraints[_norm(name)]}"
            )
            checked += 1
    return checked


def test_pyproject_pins_match_constraints(constraints):
    pins = QUOTED_PIN.findall(_read("pyproject.toml"))
    assert _assert_agrees(pins, constraints, "pyproject.toml") > 0


def test_api_requirements_pin_the_constrained_mlflow_client(constraints):
    # Dockerfile.api and the Hugging Face Space install this file on its own, so it repeats the pin.
    lines = _requirement_lines(_read("precisionai/magnidata/api/requirements.txt"))
    pins = [(m.group(1), m.group(2)) for m in map(PIN.match, lines) if m]
    assert ("mlflow-skinny", constraints["mlflow-skinny"]) in pins
    _assert_agrees(pins, constraints, "precisionai/magnidata/api/requirements.txt")


@pytest.mark.parametrize("relative", ["requirements.txt", "requirements-dev.txt"])
def test_root_requirement_files_take_shared_versions_from_constraints(constraints, relative):
    lines = _requirement_lines(_read(relative))
    assert "-c constraints.txt" in lines
    pinned_here = [m.group(1) for m in map(PIN.match, lines) if m and _norm(m.group(1)) in constraints]
    assert pinned_here == [], f"{relative} pins {pinned_here}; take the version from constraints.txt instead"


def test_requirements_dev_installs_the_package_with_its_api_dependencies():
    # The tests import Flask and the rest of [project.dependencies]; requirements.txt alone has none of them.
    assert "-e .[dev]" in _requirement_lines(_read("requirements-dev.txt"))


def test_requirements_txt_lists_the_constrained_ml_stack(constraints):
    names = {_norm(m.group(1)) for m in map(BARE.match, _requirement_lines(_read("requirements.txt"))) if m}
    assert {"torch", "torchvision", "timm", "pyiqa"} <= names
    assert {"torch", "torchvision", "timm", "pyiqa"} <= constraints.keys()


def test_mlflow_server_image_runs_the_client_release(constraints):
    assert constraints["mlflow"] == constraints["mlflow-skinny"]
    images = re.findall(r"image:\s*ghcr\.io/mlflow/mlflow:v?(\S+)", _read("docker-compose.yml"))
    assert images == [constraints["mlflow"]]


def test_pre_commit_ruff_rev_matches_the_pinned_ruff(constraints):
    config = _read(".pre-commit-config.yaml")
    match = re.search(r"repo:\s*https://github\.com/astral-sh/ruff-pre-commit\s*\n\s*rev:\s*v?(\S+)", config)
    assert match, "ruff-pre-commit repo not found in .pre-commit-config.yaml"
    assert match.group(1) == constraints["ruff"]


def _pip_install_commands(dockerfile: str) -> list[str]:
    joined = re.sub(r"\\\n", " ", dockerfile)
    return [
        cmd
        for line in joined.splitlines()
        if not line.lstrip().startswith("#")
        for cmd in line.split("&&")
        if "pip install" in cmd
    ]


def test_trainer_image_installs_everything_under_constraints(constraints):
    dockerfile = _read("Dockerfile.trainer")
    assert "COPY constraints.txt" in dockerfile
    commands = _pip_install_commands(dockerfile)
    assert len(commands) >= 2
    for command in commands:
        assert "-c constraints.txt" in command, f"Dockerfile.trainer installs without constraints: {command.strip()}"
        pinned = [name for name, _ in re.findall(r"([A-Za-z0-9._-]+)==(\S+)", command) if _norm(name) in constraints]
        assert pinned == [], f"Dockerfile.trainer pins {pinned} directly; constraints.txt holds those versions"
    torch_install = next(c for c in commands if "torch" in c.split() and "--index-url" in c)
    assert "${TORCH_INDEX_URL}" in torch_install


@pytest.mark.parametrize("workflow", ["unit-test.yml", "pre-commit.yml"])
def test_ci_installs_under_constraints(workflow):
    text = _read(f".github/workflows/{workflow}")
    installs = [line for line in text.splitlines() if "pip install" in line]
    assert installs
    assert all("-c constraints.txt" in line for line in installs)
