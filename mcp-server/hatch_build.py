# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Hatch build hook that packages ``precisionai/magnidata/mcp_server`` from the main source tree."""

from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

PACKAGE = "precisionai/magnidata/mcp_server"


class CustomBuildHook(BuildHookInterface):
    """Add the MCP server's modules, without its tests, to the sdist and the wheel."""

    def initialize(self, version: str, build_data: dict[str, Any]) -> None:
        """Force-include the package files, found in the repository checkout or in an unpacked sdist.

        Parameters
        ----------
        version : str
            Build version requested by hatchling (unused).
        build_data : dict
            Hatchling's mutable build data; ``force_include`` maps source paths to paths in the archive.
        """
        root = Path(self.root)
        # An unpacked sdist holds the package next to pyproject.toml; a checkout holds it one level up.
        source = next((base / PACKAGE for base in (root, root.parent) if (base / PACKAGE).is_dir()), None)
        if source is None:
            raise FileNotFoundError(f"{PACKAGE} not found next to or above {root}")
        for module in sorted(source.glob("*.py")):
            if not module.name.startswith("test_") and module.name != "conftest.py":
                build_data["force_include"][str(module)] = f"{PACKAGE}/{module.name}"
        license_file = root.parent / "LICENSE.md"
        if self.target_name == "sdist" and license_file.is_file():
            build_data["force_include"][str(license_file)] = "LICENSE.md"
