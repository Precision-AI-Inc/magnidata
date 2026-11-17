# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Keep the magnidata agent skill's HTTP API reference in step with the routes the Flask app serves.

Run:  python -m pytest precisionai/magnidata/api/test_agent_skill_api_reference.py -q
"""

import re
from pathlib import Path

import pytest
from conftest import load_app

REPO = Path(__file__).resolve().parents[3]
HTTP_API = REPO / "plugins" / "magnidata" / "skills" / "magnidata" / "references" / "http-api.md"
# Routes the skill deliberately leaves out: an unused debug endpoint and the dashboard's MLflow probe.
UNDOCUMENTED_ROUTES = {"GET /api/mask/debug", "GET /api/mlflow/status"}
PARAM = re.compile(r"<[^>]*>")


def _served_routes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> set[str]:
    app = load_app(tmp_path, monkeypatch).app
    return {
        f"{method} {PARAM.sub('<>', rule.rule)}"
        for rule in app.url_map.iter_rules()
        if rule.rule.startswith("/api/")
        for method in (rule.methods or set()) - {"HEAD", "OPTIONS"}
    }


def test_http_reference_covers_every_served_api_route(tmp_path, monkeypatch):
    served = _served_routes(tmp_path, monkeypatch)
    assert "GET /api/runs/<>/metrics" in served
    documented = PARAM.sub("<>", HTTP_API.read_text())
    missing = sorted(r for r in served - UNDOCUMENTED_ROUTES if f"`{r}`" not in documented)
    assert not missing, f"add these routes to {HTTP_API.relative_to(REPO)}: {missing}"


def test_undocumented_routes_are_still_served(tmp_path, monkeypatch):
    assert _served_routes(tmp_path, monkeypatch) >= UNDOCUMENTED_ROUTES
