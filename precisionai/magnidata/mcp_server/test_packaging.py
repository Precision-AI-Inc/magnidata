# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Keep mcp-server/ (PyPI metadata, registry entry, README) consistent with the code it packages."""

import importlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from precisionai.magnidata.mcp_server.client import DEFAULT_API_URL
from precisionai.magnidata.mcp_server.server import API_URL_ENV

PACKAGING = Path(__file__).resolve().parents[3] / "mcp-server"


@pytest.fixture
def server_json() -> dict[str, Any]:
    return json.loads((PACKAGING / "server.json").read_text())


@pytest.fixture
def pyproject() -> str:
    return (PACKAGING / "pyproject.toml").read_text()


def test_registry_name_is_claimed_in_the_readme(server_json: dict[str, Any]):
    readme = (PACKAGING / "README.md").read_text()
    assert f"<!-- mcp-name: {server_json['name']} -->" in readme
    assert server_json["name"] == "io.github.Precision-AI-Inc/magnidata"


def test_registry_entry_points_at_this_package(server_json: dict[str, Any], pyproject: str):
    (package,) = server_json["packages"]
    assert re.search(rf'^name = "{re.escape(package["identifier"])}"$', pyproject, re.MULTILINE)
    assert package["registryType"] == "pypi"
    assert package["runtimeHint"] == "uvx"
    assert package["transport"] == {"type": "stdio"}
    assert len(server_json["description"]) <= 100
    assert server_json["repository"]["subfolder"] == PACKAGING.name


def test_registry_entry_documents_the_api_url_variable(server_json: dict[str, Any]):
    (variable,) = server_json["packages"][0]["environmentVariables"]
    assert variable["name"] == API_URL_ENV
    assert variable["default"] == DEFAULT_API_URL


def test_console_script_resolves_to_main(pyproject: str):
    match = re.search(r'^magnidata-mcp = "([\w.]+):(\w+)"$', pyproject, re.MULTILINE)
    assert match is not None
    module, function = match.groups()
    assert callable(getattr(importlib.import_module(module), function))
