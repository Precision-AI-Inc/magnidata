# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Reproducibility sidecar: capture exactly what produced an output CSV.

Written next to the output as ``<output>.provenance.json``. Records code version,
library versions, model-weight hashes, all extraction parameters, and the input CSV
hash, so a run can be tied to — and reproduced from — a known state.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import subprocess
import sys


def sha256_file(path: str) -> str:
    """Return the hex-encoded SHA-256 digest of the file at ``path``."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _module_version(name: str) -> str:
    try:
        mod = __import__(name)
        return getattr(mod, "__version__", "?")
    except Exception:
        return "absent"


def _git() -> dict:
    """Return the current git commit hash and working-tree dirty flag, best-effort.

    Both git invocations below pass a hardcoded, fully literal argument list to
    `subprocess.run` — never built from external input — with the default
    `shell=False`, so this is a safe, fully static subprocess call rather than the
    "untrusted input" pattern security scanners warn about. The executable is the
    full path `/usr/bin/git` (this repo's CI and dev/container images are all
    Debian/Ubuntu-based) rather than a bare `git`, per the standard mitigation for
    partial-executable-path warnings; on any host where git isn't there — including
    the API container image, which doesn't install git at all — the call simply
    fails and is swallowed below exactly as it would for a bare, unresolved `git`.
    """
    try:
        commit = subprocess.run(
            ["/usr/bin/git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=False
        ).stdout.strip()
    except Exception:
        commit = ""
    commit = commit or os.environ.get("GIT_COMMIT", "")

    try:
        dirty = bool(
            subprocess.run(
                ["/usr/bin/git", "status", "--porcelain"], capture_output=True, text=True, timeout=10, check=False
            ).stdout.strip()
        )
    except Exception:
        dirty = False

    return {"commit": commit, "dirty": dirty}


DEFAULT_DETERMINISM_NOTES = [
    "float64 reductions; values rounded only at the output boundary.",
    "Polygon rasterization in COCO file order (deterministic).",
    "Colour masks (used when an image has no COCO label) decoded with near-black as background "
    "and one class per exact colour, resized nearest-neighbour to the image read.",
    "NIMA float values may drift across torch versions — pin torch for exact reproduction.",
]

_SOURCE_NAMES = {"fullres": "full-resolution", "thumb720": "720 px thumbnail", "thumb64": "64 px thumbnail"}


def image_source_note(image_sources: dict[str, int]) -> str:
    """Return a determinism note stating which image resolution the features were computed on.

    Parameters
    ----------
    image_sources : dict of str to int
        Row count per ``image_source`` value (``fullres``, ``thumb720``, ``thumb64``,
        ``missing``), as recorded in the output CSV.

    Returns
    -------
    str
        A note saying the run used full-resolution images only, or listing the thumbnail and
        missing counts, since thumbnail-derived pixel features are not comparable to
        full-resolution ones.
    """
    read = {k: v for k, v in image_sources.items() if v and k != "missing"}
    missing = image_sources.get("missing", 0)
    if set(read) <= {"fullres"} and not missing:
        return "Features computed on full-resolution images (no resize/thumbnail)."
    counts = ", ".join(f"{n} {_SOURCE_NAMES.get(k, k)}" for k, n in sorted(read.items()))
    note = f"Features computed on mixed or downscaled images: {counts or 'none read'}"
    if missing:
        note += f"; {missing} not found"
    return note + (". Thumbnail-derived pixel features are not comparable to full-resolution values.")


def build(
    input_path: str,
    output_path: str,
    params: dict,
    schema_version: str,
    weight_paths: list[str] | None = None,
    version_modules: tuple[str, ...] = ("numpy", "PIL", "torch", "pyiqa"),
    determinism_notes: list[str] | None = None,
) -> dict:
    """Assemble the provenance record for one tool run.

    ``determinism_notes`` defaults to the feature extractor's notes, which say nothing
    about image resolution: the feature extractor prepends ``image_source_note`` for the
    sources it actually read. Tools whose reproducibility caveats differ (e.g.
    embeddings.py, whose model preprocessing resizes its input and involves no COCO/NIMA
    stage) must pass their own so the sidecar describes how that output was actually
    produced.
    """
    versions = {m: _module_version(m) for m in version_modules}
    versions["python"] = sys.version.split()[0]

    weights = []
    for p in weight_paths or []:
        with contextlib.suppress(OSError):
            weights.append({"path": p, "sha256": sha256_file(p)})

    return {
        "feature_schema_version": schema_version,
        "git": _git(),
        "platform": platform.platform(),
        "versions": versions,
        "params": params,
        "model_weights": weights,
        "input_path": {"path": input_path, "sha256": sha256_file(input_path)},
        "output_path": output_path,
        "determinism_notes": (
            list(DEFAULT_DETERMINISM_NOTES) if determinism_notes is None else list(determinism_notes)
        ),
    }


def write(provenance: dict, path: str) -> None:
    """Write a provenance record as pretty-printed, key-sorted JSON to ``path``."""
    with open(path, "w") as f:
        json.dump(provenance, f, indent=2, sort_keys=True)
