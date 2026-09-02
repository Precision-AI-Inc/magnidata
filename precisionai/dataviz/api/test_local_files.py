# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for local_files.py's remote-image fetching — specifically the host
allowlist that keeps /api/image (which passes caller-supplied `path` straight through)
from becoming an open SSRF proxy.

Run:  python -m pytest precisionai/agriviz/api/test_local_files.py -q
"""
import local_files
import pytest


def test_is_remote_url():
    assert local_files.is_remote_url("https://cdn.example/a.png") is True
    assert local_files.is_remote_url("http://cdn.example/a.png") is True
    assert local_files.is_remote_url("image_sets/coco128/images/a.jpg") is False
    assert local_files.is_remote_url("/etc/passwd") is False


def test_read_file_rejects_disallowed_remote_host():
    with pytest.raises(PermissionError):
        local_files.read_file("https://evil.example/internal-metadata")


def test_read_file_fetches_allowed_remote_host(monkeypatch):
    class FakeResponse:
        def read(self):
            return b"fake-image-bytes"

        def geturl(self):
            return "https://d379glmvb7evte.cloudfront.net/a.png"

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    captured = {}

    def fake_urlopen(url, timeout=None):
        captured["url"] = url
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(local_files.urllib.request, "urlopen", fake_urlopen)

    data = local_files.read_file("https://d379glmvb7evte.cloudfront.net/a.png")

    assert data == b"fake-image-bytes"
    assert captured["url"] == "https://d379glmvb7evte.cloudfront.net/a.png"


def test_read_file_rejects_redirect_to_disallowed_host(monkeypatch):
    class FakeResponse:
        def read(self):
            return b"fake-image-bytes"

        def geturl(self):
            return "https://evil.example/a.png"   # landed somewhere else after a redirect

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(local_files.urllib.request, "urlopen", lambda url, timeout=None: FakeResponse())

    with pytest.raises(PermissionError):
        local_files.read_file("https://d379glmvb7evte.cloudfront.net/a.png")


def test_read_thumbnail_returns_none_for_remote_urls():
    assert local_files.read_thumbnail("https://d379glmvb7evte.cloudfront.net/a.png", 720) is None


def test_read_file_still_reads_local_paths(tmp_path):
    p = tmp_path / "a.bin"
    p.write_bytes(b"local-bytes")
    assert local_files.read_file(str(p)) == b"local-bytes"
