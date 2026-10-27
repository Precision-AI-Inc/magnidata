# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Load datasets ahead of time so a deployment starts with them already in place.

The Hugging Face Space image runs this once while it is built. A demo key drives the same
``POST /api/datasets/build`` endpoint the portal's "Bring Your Own Data" dialog uses and polls
the job, so the result is exactly what a click in the portal would produce: the feature CSV,
the embeddings, the precomputed projections and the database record. ``--bundle`` installs a
dataset exported by ``scripts/export_demo_bundle.py`` instead, for data too large to build
inside an image.

    python preload_demo.py --bundle /bundles/agristress500 coco128
"""

import argparse
import os
import shutil
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import db
import embeddings_io
import projections

from precisionai.magnidata.scripts import export_demo_bundle as bundles

USER_DATA_REL = "data_user"
POLL_SECONDS = 2.0
BUILD_TIMEOUT_SECONDS = 1800.0
BUILD_ATTEMPTS = 3
RETRY_DELAY_SECONDS = 10.0


class ApiClient(Protocol):
    """The part of Flask's test client this module uses, so a test can pass a scripted stand-in."""

    def get(self, url: str) -> Any:
        """Send a GET request."""
        ...

    def post(self, url: str, data: Any = None) -> Any:
        """Send a POST request with form data."""
        ...


class PreloadError(RuntimeError):
    """Raised when a demo dataset could not be built."""


def wait_for_job(
    client: ApiClient,
    job_id: str,
    *,
    timeout: float = BUILD_TIMEOUT_SECONDS,
    poll_seconds: float = POLL_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Poll a build job until it is done or has failed, and return its final state.

    Parameters
    ----------
    client : ApiClient
        Test client for the dashboard API.
    job_id : str
        Id returned by ``POST /api/datasets/build``.
    timeout : float
        Seconds to wait before giving up on a job that is still running.
    poll_seconds : float
        Seconds to sleep between status requests.
    sleep : Callable[[float], None]
        Sleep function, replaceable in tests.
    clock : Callable[[], float]
        Monotonic clock, replaceable in tests.

    Returns
    -------
    dict[str, Any]
        The job, whose ``status`` is ``"done"`` or ``"error"``.

    Raises
    ------
    PreloadError
        If the job is unknown or still running when ``timeout`` passes.
    """
    deadline = clock() + timeout
    while True:
        response = client.get(f"/api/datasets/build/{job_id}")
        if response.status_code != 200:
            raise PreloadError(f"build job {job_id} not found (HTTP {response.status_code})")
        job = response.get_json()
        if job["status"] in {"done", "error"}:
            return job
        if clock() >= deadline:
            raise PreloadError(f"build job {job_id} still {job['status']!r} after {timeout:.0f}s")
        sleep(poll_seconds)


def build_demo(
    client: ApiClient,
    demo_key: str,
    *,
    attempts: int = BUILD_ATTEMPTS,
    retry_delay: float = RETRY_DELAY_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
    **wait_options: Any,
) -> dict[str, Any]:
    """Build one demo dataset, retrying failures such as a dropped download.

    Parameters
    ----------
    client : ApiClient
        Test client for the dashboard API.
    demo_key : str
        Key of an enabled entry in ``GET /api/datasets/demos``, for example ``"coco128"``.
    attempts : int
        How many times to try before giving up.
    retry_delay : float
        Seconds to wait between attempts.
    sleep : Callable[[float], None]
        Sleep function, replaceable in tests.
    **wait_options : Any
        Passed to :func:`wait_for_job`.

    Returns
    -------
    dict[str, Any]
        The registered dataset, as reported by the finished job.

    Raises
    ------
    PreloadError
        If the build cannot be started or still fails after ``attempts`` tries.
    """
    last_error = "unknown error"
    for attempt in range(1, attempts + 1):
        response = client.post("/api/datasets/build", data={"source_type": "demo", "demo_key": demo_key})
        if response.status_code != 202:
            detail = (response.get_json() or {}).get("error", f"HTTP {response.status_code}")
            raise PreloadError(f"could not start the {demo_key!r} build: {detail}")
        job = wait_for_job(client, response.get_json()["job_id"], sleep=sleep, **wait_options)
        if job["status"] == "done":
            return job["dataset"]
        last_error = job.get("error") or job.get("message") or last_error
        print(f"{demo_key}: attempt {attempt}/{attempts} failed: {last_error}", file=sys.stderr, flush=True)
        if attempt < attempts:
            sleep(retry_delay)
    raise PreloadError(f"the {demo_key!r} build failed after {attempts} attempts: {last_error}")


def _write_projections(csv_path: Path, embeddings_path: Path) -> None:
    """Precompute the 3D projections beside the table, as the portal's own dataset builds do."""
    try:
        matrix = embeddings_io.matrix_for_csv(str(csv_path), str(embeddings_path))
        positions = projections.compute(matrix)
        if positions:
            projections.write(projections.sidecar_path(str(csv_path)), matrix.shape[0], positions)
    except Exception as exc:  # a projection is a speed-up: the viewer computes it on demand without one
        print(f"projections skipped for {csv_path.name}: {exc}", file=sys.stderr)


def install_bundle(bundle_dir: Path, data_root: Path | None = None) -> dict[str, Any]:
    """Install a dataset exported by ``export_demo_bundle`` and register it with the dashboard.

    The files land where a built dataset's would (``data_user/<stem>.csv``, ``<stem>.json`` and
    ``<stem>_images/``) and the record has the same shape a demo build registers: a first-class
    dataset with its own embeddings. A dataset already registered under that source is replaced.

    Parameters
    ----------
    bundle_dir : Path
        A directory written by ``export_demo_bundle``.
    data_root : Path | None
        The deployment's ``DATA_ROOT``; defaults to the ``DATA_ROOT`` environment variable.

    Returns
    -------
    dict[str, Any]
        The registered dataset record.

    Raises
    ------
    BundleError
        If the bundle is incomplete or inconsistent.
    """
    root = Path(os.environ.get("DATA_ROOT", ".") if data_root is None else data_root)
    manifest = bundles.validate_bundle(bundle_dir)
    stem = manifest["stem"]
    source = f"{USER_DATA_REL}/{stem}.csv"
    embeddings = f"{USER_DATA_REL}/{stem}.json"

    (root / USER_DATA_REL).mkdir(parents=True, exist_ok=True)
    shutil.copyfile(bundle_dir / f"{stem}.csv", root / source)
    shutil.copyfile(bundle_dir / f"{stem}.json", root / embeddings)
    shutil.copytree(bundle_dir / f"{stem}_images", root / USER_DATA_REL / f"{stem}_images", dirs_exist_ok=True)
    _write_projections(root / source, root / embeddings)

    db.init_db()
    existing = db.get_created_dataset_by_source(source)
    if existing is not None:
        db.delete_created_dataset(existing["id"])
    return db.create_dataset_record(
        manifest["name"], manifest["description"], source, None, embeddings, manifest["row_count"]
    )


def main(client: ApiClient, demo_keys: Sequence[str], bundle_dirs: Sequence[Path] = ()) -> int:
    """Install every bundle and build every named demo dataset, and return a process exit code.

    Parameters
    ----------
    client : ApiClient
        Test client for the dashboard API.
    demo_keys : Sequence[str]
        Demo keys to build, in order.
    bundle_dirs : Sequence[Path]
        Bundles to install, before the demos are built.

    Returns
    -------
    int
        0 when everything was loaded, 1 when something failed, 2 when nothing was asked for.
    """
    if not demo_keys and not bundle_dirs:
        print("usage: python preload_demo.py [--bundle DIR ...] [<demo-key> ...]", file=sys.stderr)
        return 2
    try:
        for bundle_dir in bundle_dirs:
            print(f"installed {bundle_dir.name}: {install_bundle(bundle_dir)}", flush=True)
        for key in demo_keys:
            print(f"preloaded {key}: {build_demo(client, key)}", flush=True)
    except (PreloadError, bundles.BundleError, OSError) as exc:
        print(f"preload failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    # Imported here, not at the top: importing app creates the database and the data folders,
    # which a test importing this module for its helpers must not do.
    from app import app as flask_app

    parser = argparse.ArgumentParser(description="Load datasets into this deployment ahead of time.")
    parser.add_argument("demo_keys", nargs="*", help="demo datasets to build in the app, for example coco128")
    parser.add_argument("--bundle", action="append", default=[], type=Path, help="bundle directory to install")
    arguments = parser.parse_args()
    sys.exit(main(flask_app.test_client(), arguments.demo_keys, arguments.bundle))
