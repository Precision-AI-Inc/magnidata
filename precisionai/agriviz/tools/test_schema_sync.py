# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Enforce that the dashboard's column-description data (descriptions.json +
colDescs.ts's EXTRA map) documents every column tools/schema.py's extractor
currently emits, with no stale leftovers from a previous schema version (bar the
runtime-derived columns allowlisted below). Pure file I/O + regex/JSON parsing:
no torch/network."""
import json
import os
import re

from .schema import OUTPUT_COLUMNS

# Columns useCSV.ts derives at runtime from the loaded CSV (e.g. parsed from the
# image filename) rather than columns tools/features.py's extractor writes —
# legitimately documented even though they will never appear in OUTPUT_COLUMNS.
DERIVED_COLUMNS = {"camera"}

AGRIVIZ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DESCRIPTIONS_JSON = os.path.join(AGRIVIZ_ROOT, "dashboard", "src", "data", "descriptions.json")
COL_DESCS_TS = os.path.join(AGRIVIZ_ROOT, "dashboard", "src", "data", "colDescs.ts")


def _descriptions_keys() -> set[str]:
    with open(DESCRIPTIONS_JSON) as f:
        data = json.load(f)
    return set(data["columns"].keys())


def _extra_keys() -> set[str]:
    with open(COL_DESCS_TS) as f:
        text = f.read()
    start = text.index("const EXTRA: Record<string, ColDesc> = {")
    end = text.index("export const COL_DESCS", start)
    body = text[start:end]
    return set(re.findall(r"^\s{2}(\w+):\s*\{", body, re.MULTILINE))


def test_dashboard_column_descriptions_match_current_schema():
    documented = _descriptions_keys() | _extra_keys()
    expected = set(OUTPUT_COLUMNS)
    missing = expected - documented
    stale = documented - expected - DERIVED_COLUMNS
    assert not missing and not stale, (
        f"schema/description drift — missing descriptions for: {sorted(missing)}; "
        f"stale descriptions (not in current schema) for: {sorted(stale)}"
    )
