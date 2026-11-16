# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Guard the agent skills and the Claude Code plugin marketplace shipped at the repository root.

`.claude-plugin/marketplace.json` lists the plugin in `plugins/magnidata/`, whose `skills/` hold the Agent Skills
that `.agents/skills/` links to for Codex. These tests check the manifests against the documented formats and keep
the skills' column reference in step with the feature schema. `api/test_agent_skill_api_reference.py` does the
same for the HTTP API reference.

Run:  python -m pytest precisionai/magnidata/test_agent_plugin.py -q
"""

import json
import re
from pathlib import Path

import pytest
import yaml

from precisionai.magnidata.tools.schema import OUTPUT_COLUMNS

REPO = Path(__file__).resolve().parents[2]
MARKETPLACE = REPO / ".claude-plugin" / "marketplace.json"
PLUGIN = REPO / "plugins" / "magnidata"
SKILLS = PLUGIN / "skills"
MIRROR = REPO / ".agents" / "skills"
COLUMNS = SKILLS / "magnidata" / "references" / "columns.md"

# Agent Skills specification (https://agentskills.io/specification): lowercase letters, digits and single hyphens.
SKILL_NAME = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
# The frontmatter keys every Agent Skills client accepts; claude.ai uploads reject anything else.
PORTABLE_KEYS = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
RESERVED_PLUGIN_PREFIXES = ("claude-", "anthropic-", "anthropics-", "cc-plugin-")


def _skill_dirs():
    return sorted(p for p in SKILLS.iterdir() if p.is_dir())


def _frontmatter(skill_md):
    text = skill_md.read_text()
    match = re.match(r"^---\n(.*?)\n---\n", text, re.DOTALL)
    assert match, f"{skill_md} has no YAML frontmatter"
    return yaml.safe_load(match.group(1)), text[match.end() :]


@pytest.fixture
def marketplace():
    return json.loads(MARKETPLACE.read_text())


def test_marketplace_has_the_required_fields(marketplace):
    assert SKILL_NAME.match(marketplace["name"])
    assert marketplace["owner"]["name"]
    assert marketplace["plugins"]


def test_every_listed_plugin_resolves_to_a_manifest_with_the_same_name(marketplace):
    for entry in marketplace["plugins"]:
        source = entry["source"]
        assert source.startswith("./")
        assert ".." not in source
        manifest = json.loads((REPO / source / ".claude-plugin" / "plugin.json").read_text())
        assert manifest["name"] == entry["name"]
        assert SKILL_NAME.match(entry["name"])
        assert not entry["name"].startswith(RESERVED_PLUGIN_PREFIXES)


def test_plugin_version_is_left_to_the_commit(marketplace):
    # Without a version, Claude Code versions the plugin by commit SHA, so every merged change reaches users.
    manifest = json.loads((PLUGIN / ".claude-plugin" / "plugin.json").read_text())
    assert "version" not in manifest
    assert all("version" not in entry for entry in marketplace["plugins"])


@pytest.mark.parametrize("skill_dir", _skill_dirs(), ids=lambda p: p.name)
def test_skill_frontmatter_follows_the_agent_skills_spec(skill_dir):
    meta, body = _frontmatter(skill_dir / "SKILL.md")
    assert set(meta) <= PORTABLE_KEYS
    assert meta["name"] == skill_dir.name
    assert SKILL_NAME.match(meta["name"])
    assert len(meta["name"]) <= 64
    assert 0 < len(meta["description"]) <= 1024
    assert len(meta.get("compatibility", "x")) <= 500
    assert len(body.splitlines()) < 500


@pytest.mark.parametrize("skill_dir", _skill_dirs(), ids=lambda p: p.name)
def test_skill_reference_files_exist(skill_dir):
    _, body = _frontmatter(skill_dir / "SKILL.md")
    for ref in re.findall(r"`(references/[^`]+)`", body):
        assert (skill_dir / ref).is_file(), ref


@pytest.mark.parametrize("skill_dir", _skill_dirs(), ids=lambda p: p.name)
def test_every_skill_is_linked_for_codex(skill_dir):
    link = MIRROR / skill_dir.name
    assert link.is_symlink()
    assert link.resolve() == skill_dir.resolve()


def test_the_codex_folder_holds_only_links_to_plugin_skills():
    assert {p.name for p in MIRROR.iterdir()} == {p.name for p in _skill_dirs()}


def test_column_reference_covers_every_feature_column():
    documented = COLUMNS.read_text()
    missing = [c for c in OUTPUT_COLUMNS if f"`{c}`" not in documented]
    assert not missing, f"add these columns to {COLUMNS.relative_to(REPO)}: {missing}"
