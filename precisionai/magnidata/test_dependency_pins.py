# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Keep every copy of a pinned dependency version in step with ``constraints.txt``.

``constraints.txt`` at the repository root holds the exact versions that more than one install path needs: the
trainer image, the ``pyproject.toml`` extras, ``requirements.txt``, the API's own requirements file, CI, the MLflow
server image in ``docker-compose.yml`` and the ruff pre-commit hook. Files that pip reads at install time use it as a
constraints file. The others cannot read it, so these tests fail when one of them names a different version, as
happened when the trainer image installed one torch build and the ``training`` extra replaced it with another.

Requirement strings are parsed with ``packaging`` (the parser pip uses), YAML with ``yaml.safe_load`` and
``pyproject.toml`` with ``tomllib``. Dockerfiles have no parser here: their ``RUN`` instructions are split into shell
commands with ``shlex``, which covers the ``cmd && cmd`` and ``if ...; then ...; fi`` forms the trainer image uses.

Run:  python -m pytest precisionai/magnidata/test_dependency_pins.py -q
"""

import re
import shlex
from pathlib import Path
from typing import Any

import pytest
import yaml
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

try:
    import tomllib as _tomllib_module

    _tomllib: Any = _tomllib_module
    _TOMLLIB_AVAILABLE = True
except ImportError:
    _tomllib = None
    _TOMLLIB_AVAILABLE = False

REPO = Path(__file__).resolve().parents[2]
OPTIONS_WITH_VALUE = {"-c", "--constraint", "-r", "--requirement", "-i", "--index-url", "--extra-index-url"}


def _read(relative: str) -> str:
    path = REPO / relative
    if not path.is_file():
        pytest.skip(f"{relative} is not part of this checkout")
    return path.read_text()


def _requirement_file(relative: str) -> tuple[list[str], list[Requirement]]:
    """Return the option lines (``-c ...``, ``-e ...``) and the parsed requirements of a pip requirements file."""
    options, requirements = [], []
    for raw in _read(relative).splitlines():
        line = re.split(r"(?:^|\s)#", raw, maxsplit=1)[0].strip()
        if line.startswith("-"):
            options.append(line)
        elif line:
            requirements.append(Requirement(line))
    return options, requirements


def _exact_version(requirement: Requirement) -> str | None:
    specifiers = list(requirement.specifier)
    if len(specifiers) == 1 and specifiers[0].operator == "==":
        return specifiers[0].version
    return None


@pytest.fixture(scope="module")
def constraints() -> dict[str, str]:
    options, requirements = _requirement_file("constraints.txt")
    assert options == []
    pins = {}
    for requirement in requirements:
        version = _exact_version(requirement)
        assert version, f"constraints.txt entry is not an exact pin: {requirement}"
        pins[canonicalize_name(requirement.name)] = version
    return pins


def _assert_agrees(requirements: list[Requirement], constraints: dict[str, str], where: str) -> int:
    checked = 0
    for requirement in requirements:
        name = canonicalize_name(requirement.name)
        version = _exact_version(requirement)
        if name in constraints and version is not None:
            assert version == constraints[name], (
                f"{where} pins {requirement}, constraints.txt has {name}=={constraints[name]}"
            )
            checked += 1
    return checked


def _pip_install_args(script: str) -> list[list[str]]:
    """Return the arguments after ``pip install`` for each such command in a shell script."""
    installs = []
    for line in script.replace("\\\n", " ").splitlines():
        for command in re.split(r"&&|\|\||;", line):
            tokens = shlex.split(command, comments=True)
            for i in range(len(tokens) - 1):
                if tokens[i : i + 2] == ["pip", "install"]:
                    installs.append(tokens[i + 2 :])
                    break
    return installs


def _option_values(args: list[str], *names: str) -> list[str]:
    return [args[i + 1] for i in range(len(args) - 1) if args[i] in names]


def _install_requirements(args: list[str]) -> list[Requirement]:
    """Parse the requirement arguments of a ``pip install``; local paths such as ``.[training]`` are skipped."""
    requirements = []
    skip_next = False
    for arg in args:
        if skip_next or arg.startswith("-"):
            skip_next = arg in OPTIONS_WITH_VALUE
            continue
        try:
            requirements.append(Requirement(arg))
        except InvalidRequirement:
            continue
    return requirements


def test_pyproject_pins_match_constraints(constraints: dict[str, str]):
    if not _TOMLLIB_AVAILABLE:
        pytest.skip("tomllib needs Python 3.11+")
    project = _tomllib.loads(_read("pyproject.toml"))["project"]
    strings = list(project["dependencies"])
    for extra in project["optional-dependencies"].values():
        strings.extend(extra)
    assert _assert_agrees([Requirement(s) for s in strings], constraints, "pyproject.toml") > 0


def test_api_requirements_pin_the_constrained_mlflow_client(constraints: dict[str, str]):
    # Dockerfile.api and the Hugging Face Space install this file on its own, so it repeats the pin.
    _, requirements = _requirement_file("precisionai/magnidata/api/requirements.txt")
    pins: dict[str, str | None] = {canonicalize_name(r.name): _exact_version(r) for r in requirements}
    assert pins.get("mlflow-skinny") == constraints["mlflow-skinny"]
    _assert_agrees(requirements, constraints, "precisionai/magnidata/api/requirements.txt")


@pytest.mark.parametrize("relative", ["requirements.txt", "requirements-dev.txt"])
def test_root_requirement_files_take_shared_versions_from_constraints(constraints: dict[str, str], relative: str):
    options, requirements = _requirement_file(relative)
    assert "-c constraints.txt" in options
    pinned_here = [str(r) for r in requirements if canonicalize_name(r.name) in constraints and r.specifier]
    assert pinned_here == [], f"{relative} pins {pinned_here}; take the version from constraints.txt instead"


def test_requirements_dev_installs_the_package_with_its_api_dependencies():
    # The tests import Flask and the rest of [project.dependencies]; requirements.txt alone has none of them.
    options, _ = _requirement_file("requirements-dev.txt")
    assert "-e .[dev]" in options


def test_requirements_txt_lists_the_constrained_ml_stack(constraints: dict[str, str]):
    _, requirements = _requirement_file("requirements.txt")
    names = {canonicalize_name(r.name) for r in requirements}
    assert {"pillow", "torch", "torchvision", "timm", "pyiqa"} <= names
    assert {"pillow", "torch", "torchvision", "timm", "pyiqa"} <= constraints.keys()


def test_mlflow_server_image_runs_the_client_release(constraints: dict[str, str]):
    assert constraints["mlflow"] == constraints["mlflow-skinny"]
    compose = yaml.safe_load(_read("docker-compose.yml"))
    repository, _, tag = compose["services"]["mlflow"]["image"].rpartition(":")
    assert repository == "ghcr.io/mlflow/mlflow"
    assert tag.removeprefix("v") == constraints["mlflow"]


def test_pre_commit_ruff_rev_matches_the_pinned_ruff(constraints: dict[str, str]):
    config = yaml.safe_load(_read(".pre-commit-config.yaml"))
    revs = [repo["rev"] for repo in config["repos"] if repo["repo"] == "https://github.com/astral-sh/ruff-pre-commit"]
    assert len(revs) == 1, "expected one ruff-pre-commit repo in .pre-commit-config.yaml"
    assert revs[0].removeprefix("v") == constraints["ruff"]


def _dockerfile_instructions(dockerfile: str, keyword: str) -> list[str]:
    joined = dockerfile.replace("\\\n", " ")
    return [line.split(None, 1)[1] for line in joined.splitlines() if line.split(None, 1)[:1] == [keyword]]


def test_trainer_image_installs_everything_under_constraints(constraints: dict[str, str]):
    dockerfile = _read("Dockerfile.trainer")
    copied = [shlex.split(args)[:-1] for args in _dockerfile_instructions(dockerfile, "COPY")]
    assert any("constraints.txt" in sources for sources in copied)
    installs = [args for body in _dockerfile_instructions(dockerfile, "RUN") for args in _pip_install_args(body)]
    assert len(installs) >= 2
    for args in installs:
        assert "constraints.txt" in _option_values(args, "-c", "--constraint"), (
            f"Dockerfile.trainer installs without constraints: pip install {shlex.join(args)}"
        )
        pinned = [
            str(r) for r in _install_requirements(args) if canonicalize_name(r.name) in constraints and r.specifier
        ]
        assert pinned == [], f"Dockerfile.trainer pins {pinned} directly; constraints.txt holds those versions"
    torch_installs = [args for args in installs if "torch" in {r.name for r in _install_requirements(args)}]
    assert len(torch_installs) == 1
    assert _option_values(torch_installs[0], "--index-url", "-i") == ["${TORCH_INDEX_URL}"]


@pytest.mark.parametrize("workflow", ["unit-test.yml", "pre-commit.yml"])
def test_ci_installs_under_constraints(workflow: str):
    config = yaml.safe_load(_read(f".github/workflows/{workflow}"))
    scripts = [step["run"] for job in config["jobs"].values() for step in job.get("steps", []) if "run" in step]
    installs = [args for script in scripts for args in _pip_install_args(script)]
    assert installs
    for args in installs:
        assert "constraints.txt" in _option_values(args, "-c", "--constraint")
