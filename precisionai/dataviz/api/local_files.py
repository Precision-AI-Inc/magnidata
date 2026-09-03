# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Local filesystem access — replaces ssh_client now that data lives on disk.

Also fetches images from a small allowlist of remote CDN hosts, for datasets whose
image_path columns are full URLs rather than local paths (e.g. AgriStress-500, served
from CloudFront) — path is caller-supplied via /api/image?path=..., so remote fetches
are restricted to ALLOWED_REMOTE_HOSTS to avoid turning this into an open SSRF proxy.
"""

import os
import urllib.parse
import urllib.request
from collections.abc import Iterator

DATALAKE_ROOT = os.environ.get("DATALAKE_ROOT", "/mnt/disks/ssd-raid/datalake-clone")
THUMB_ROOT_64 = os.environ.get("THUMB_ROOT_64", "/mnt/disks/ssd-raid/datalake-clone-thumbnails/64")
THUMB_ROOT_720 = os.environ.get("THUMB_ROOT_720", "/mnt/disks/ssd-raid/datalake-clone-thumbnails/720")

# Comma-separated hostnames this API is allowed to fetch remote images from. Defaults
# to the AgriStress-500 CDN; extend via env if another dataset adds a different host.
ALLOWED_REMOTE_HOSTS = {
    h.strip()
    for h in os.environ.get(
        "ALLOWED_REMOTE_IMAGE_HOSTS",
        "d379glmvb7evte.cloudfront.net",
    ).split(",")
    if h.strip()
}


def is_remote_url(path: str) -> bool:
    """Return whether path is an http(s) URL rather than a local filesystem path."""
    return path.startswith("http://") or path.startswith("https://")


def _remote_host_allowed(url: str) -> bool:
    return (urllib.parse.urlparse(url).hostname or "") in ALLOWED_REMOTE_HOSTS


def _thumb_candidates(path: str, thumb_root: str) -> Iterator[str]:
    """Thumbnail path candidates for a datalake image.

    Thumbnails mirror the datalake layout but are stored as .jpg (sources are
    .png), so try .jpg/.jpeg first, then the original extension.
    """
    if not path.startswith(DATALAKE_ROOT):
        return
    rel = path[len(DATALAKE_ROOT) :]
    base, ext = os.path.splitext(rel)
    for e in (".jpg", ".jpeg", ext, ".png"):
        yield thumb_root + base + e


def read_file(path: str) -> bytes:
    """Read a local file's bytes, or fetch them from an allowlisted remote host."""
    if is_remote_url(path):
        if not _remote_host_allowed(path):
            raise PermissionError(f"remote host not allowed: {path}")
        if urllib.parse.urlparse(path).scheme not in ("http", "https"):
            raise ValueError(f"unsupported URL scheme: {path!r}")
        with urllib.request.urlopen(path, timeout=20) as resp:
            if not _remote_host_allowed(resp.geturl()):  # guard against redirect bypass
                raise PermissionError("redirected to a disallowed host")
            return resp.read()
    with open(path, "rb") as f:
        return f.read()


def read_thumbnail(path: str, max_size: int) -> bytes | None:
    """Return pre-generated thumbnail bytes, or None if none exist for this size.

    Remote sources have no local pre-generated thumbnail — callers pass whichever URL
    (full-res or a dataset's own thumbnail column) they want served, and read_file
    fetches it directly.
    """
    if is_remote_url(path):
        return None
    if max_size <= 64:
        root = THUMB_ROOT_64
    elif max_size <= 720:
        root = THUMB_ROOT_720
    else:
        return None
    for thumb in _thumb_candidates(path, root):
        if os.path.isfile(thumb):
            with open(thumb, "rb") as f:
                return f.read()
    return None


def local_status() -> dict:
    """Return whether the datalake root is accessible, and its configured path."""
    return {
        "datalake_accessible": os.path.isdir(DATALAKE_ROOT),
        "root": DATALAKE_ROOT,
    }
