# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for preload_demo: polling a build job, retrying a failed build, and the exit codes.

The dashboard API is replaced by a scripted client — the build pipeline itself is covered by
test_dataset_builder.py, and the real round trip by building the Space image.

Run:  python -m pytest precisionai/magnidata/api/test_preload_demo.py -q
"""

import csv
import io
import json
import re

import preload_demo
import pytest
from conftest import load_app

from precisionai.magnidata.scripts import export_demo_bundle as bundles


class _Response:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def get_json(self):
        return self._body


class _ScriptedClient:
    """Answers POSTs and job polls from fixed lists, and records what was asked."""

    def __init__(self, posts, jobs):
        self._posts = list(posts)
        self._jobs = list(jobs)
        self.post_calls = []
        self.get_calls = []

    def post(self, url, data=None):
        self.post_calls.append((url, data))
        return self._posts.pop(0)

    def get(self, url):
        self.get_calls.append(url)
        return self._jobs.pop(0)


def _started(job_id="job-1"):
    return _Response(202, {"job_id": job_id})


def _job(status, **fields):
    return _Response(200, {"status": status, "error": None, "message": "", "dataset": None, **fields})


def _no_sleep(_seconds):
    return None


def test_wait_for_job_polls_until_done():
    client = _ScriptedClient([], [_job("staging"), _job("extracting_features"), _job("done", dataset={"id": 1})])
    sleeps = []

    job = preload_demo.wait_for_job(client, "job-1", sleep=sleeps.append, poll_seconds=0.5)

    assert job["status"] == "done"
    assert job["dataset"] == {"id": 1}
    assert sleeps == [0.5, 0.5]
    assert client.get_calls == ["/api/datasets/build/job-1"] * 3


def test_wait_for_job_returns_a_failed_job():
    client = _ScriptedClient([], [_job("error", error="boom")])

    job = preload_demo.wait_for_job(client, "job-1", sleep=_no_sleep)

    assert job["status"] == "error"
    assert job["error"] == "boom"


def test_wait_for_job_gives_up_after_the_timeout():
    client = _ScriptedClient([], [_job("staging")] * 5)
    ticks = iter([0.0, 1.0, 2.0, 3.0, 4.0])

    with pytest.raises(preload_demo.PreloadError, match="still 'staging' after 2s"):
        preload_demo.wait_for_job(client, "job-1", timeout=2.0, sleep=_no_sleep, clock=lambda: next(ticks))


def test_wait_for_job_rejects_an_unknown_job():
    client = _ScriptedClient([], [_Response(404, {"error": "job not found"})])

    with pytest.raises(preload_demo.PreloadError, match="job-1 not found"):
        preload_demo.wait_for_job(client, "job-1", sleep=_no_sleep)


def test_build_demo_posts_the_demo_form_and_returns_the_dataset():
    client = _ScriptedClient([_started()], [_job("done", dataset={"name": "COCO128"})])

    dataset = preload_demo.build_demo(client, "coco128", sleep=_no_sleep)

    assert dataset == {"name": "COCO128"}
    assert client.post_calls == [("/api/datasets/build", {"source_type": "demo", "demo_key": "coco128"})]


def test_build_demo_retries_a_failed_build():
    client = _ScriptedClient(
        [_started("job-1"), _started("job-2")],
        [_job("error", error="download dropped"), _job("done", dataset={"name": "COCO128"})],
    )
    sleeps = []

    dataset = preload_demo.build_demo(client, "coco128", retry_delay=7.0, sleep=sleeps.append)

    assert dataset == {"name": "COCO128"}
    assert len(client.post_calls) == 2
    assert sleeps == [7.0]


def test_build_demo_reports_the_last_error_when_every_attempt_fails():
    client = _ScriptedClient(
        [_started(), _started()],
        [_job("error", error="first"), _job("error", error="second")],
    )

    with pytest.raises(preload_demo.PreloadError, match="failed after 2 attempts: second"):
        preload_demo.build_demo(client, "coco128", attempts=2, sleep=_no_sleep)


def test_build_demo_does_not_retry_a_build_that_cannot_start():
    client = _ScriptedClient([_Response(400, {"error": "unknown or disabled demo 'nope'"})], [])

    with pytest.raises(preload_demo.PreloadError, match="unknown or disabled demo 'nope'"):
        preload_demo.build_demo(client, "nope", sleep=_no_sleep)

    assert len(client.post_calls) == 1


def test_main_builds_every_key_and_exits_zero(capsys):
    client = _ScriptedClient(
        [_started("a"), _started("b")],
        [_job("done", dataset={"name": "one"}), _job("done", dataset={"name": "two"})],
    )

    assert preload_demo.main(client, ["coco128", "agristress500"]) == 0

    out = capsys.readouterr().out
    assert "preloaded coco128" in out
    assert "preloaded agristress500" in out


def test_main_exits_nonzero_when_a_build_fails(capsys, monkeypatch):
    monkeypatch.setattr(preload_demo.time, "sleep", _no_sleep)
    client = _ScriptedClient([_Response(400, {"error": "disabled"})], [])

    assert preload_demo.main(client, ["coco128"]) == 1

    assert "preload failed" in capsys.readouterr().err


def test_main_without_keys_prints_usage(capsys):
    assert preload_demo.main(_ScriptedClient([], []), []) == 2

    assert "usage" in capsys.readouterr().err


# ── Installing an exported bundle ────────────────────────────────────────────────


@pytest.fixture
def bundle_dir(synthetic_dataset, tmp_path):
    """A bundle exported from the synthetic dataset, with images capped at 16 px."""
    table = {name: [float(i), 1.0, 0.5] for i, name in enumerate(synthetic_dataset.names)}
    (synthetic_dataset.root / "data_user" / "syn.json").write_text(json.dumps({"embeddings": table}))
    out = tmp_path / "bundle"
    bundles.export_bundle(
        synthetic_dataset.root, "syn", out, name="Synthetic", description="eight tiles",
        max_size=16, thumbnail_size=8, workers=1,
    )  # fmt: skip
    return out


@pytest.fixture
def deployment(tmp_path, monkeypatch):
    """A fresh deployment: app loaded against an empty DATA_ROOT, with that as the working directory."""
    root = tmp_path / "deployment"
    app_module = load_app(root, monkeypatch)
    monkeypatch.chdir(root)  # image paths in the table are relative to DATA_ROOT, as in the container
    return root, app_module


def test_an_installed_bundle_is_served_by_the_app(bundle_dir, deployment):
    root, app_module = deployment

    record = preload_demo.install_bundle(bundle_dir, data_root=root)

    assert record["name"] == "Synthetic"
    assert record["source"] == "data_user/syn.csv"
    assert record["parent_source"] is None
    assert record["row_count"] == 8
    with app_module.app.test_client() as client:
        listed = {d["name"]: d for d in client.get("/api/datasets").get_json()}
        assert listed["Synthetic"]["has_embeddings"] is True
        table = client.get("/api/dataset/csv", query_string={"source": "data_user/syn.csv"})
        assert table.status_code == 200
        first = next(csv.DictReader(io.StringIO(table.get_data(as_text=True))))
        image = client.get("/api/image", query_string={"path": first["image_path"]})
        assert (image.status_code, image.mimetype) == (200, "image/jpeg")
        overlay = client.get("/api/annotation/overlay", query_string={"path": first["image_path"]})
        assert (overlay.status_code, overlay.mimetype) == (200, "image/jpeg")
        embeddings = client.get("/api/dataset/embeddings", query_string={"source": "data_user/syn.csv"})
        assert embeddings.status_code == 200


def test_installing_twice_keeps_one_record(bundle_dir, deployment):
    root, app_module = deployment

    preload_demo.install_bundle(bundle_dir, data_root=root)
    preload_demo.install_bundle(bundle_dir, data_root=root)

    with app_module.app.test_client() as client:
        assert [d["name"] for d in client.get("/api/datasets").get_json()] == ["Synthetic"]


def test_data_root_defaults_to_the_environment(bundle_dir, deployment, monkeypatch):
    root, _ = deployment
    monkeypatch.setenv("DATA_ROOT", str(root))

    preload_demo.install_bundle(bundle_dir)

    assert (root / "data_user" / "syn.csv").is_file()


def test_an_incomplete_bundle_is_not_installed(bundle_dir, deployment):
    root, _ = deployment
    (bundle_dir / "syn_images" / "images" / "img_00.jpg").unlink()

    with pytest.raises(bundles.BundleError, match=re.escape("missing data_user/syn_images/images/img_00.jpg")):
        preload_demo.install_bundle(bundle_dir, data_root=root)

    assert not (root / "data_user" / "syn.csv").exists()


def test_failing_projections_do_not_stop_an_install(bundle_dir, deployment, monkeypatch, capsys):
    root, _ = deployment

    def broken(*_args, **_kwargs):
        raise RuntimeError("no memory")

    monkeypatch.setattr(preload_demo.projections, "compute", broken)

    record = preload_demo.install_bundle(bundle_dir, data_root=root)

    assert record["name"] == "Synthetic"
    assert "projections skipped for syn.csv: no memory" in capsys.readouterr().err


def test_main_installs_bundles_before_building_demos(bundle_dir, deployment, monkeypatch, capsys):
    root, _ = deployment
    monkeypatch.setenv("DATA_ROOT", str(root))
    client = _ScriptedClient([_started()], [_job("done", dataset={"name": "COCO128"})])

    assert preload_demo.main(client, ["coco128"], [bundle_dir]) == 0

    out = capsys.readouterr().out
    assert out.index("installed bundle") < out.index("preloaded coco128")


def test_main_reports_a_bad_bundle_without_building_anything(deployment, tmp_path, capsys):
    client = _ScriptedClient([], [])

    assert preload_demo.main(client, ["coco128"], [tmp_path]) == 1

    assert "preload failed" in capsys.readouterr().err
    assert client.post_calls == []


def test_main_with_only_a_bundle_needs_no_demo_key(bundle_dir, deployment, monkeypatch):
    root, _ = deployment
    monkeypatch.setenv("DATA_ROOT", str(root))

    assert preload_demo.main(_ScriptedClient([], []), [], [bundle_dir]) == 0
