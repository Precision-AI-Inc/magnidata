# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Guard the order of steps in the Hugging Face Space Dockerfile.

The image build loads the datasets through the same endpoint demo mode refuses, so demo mode may only
be switched on after that step; set earlier, the build fails with "Disabled for the Demo".

Run:  python -m pytest precisionai/magnidata/api/test_space_dockerfile.py -q
"""

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
