# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Trainer worker: claims ``SCHEDULED`` UI-launched MLflow runs and trains them one at a time.

MLflow is the job queue. MLflow has no compare-and-set, so version 1 supports exactly one
worker; the tag-and-re-read claim below only protects against accidental double starts.
Training runs in a subprocess so a CUDA or out-of-memory crash cannot take the worker down.

Usage::

    python -m precisionai.magnidata.training.worker --data-root /app
"""

import argparse
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from precisionai.magnidata.training import contract

try:
    import mlflow
except ImportError:
    raise ImportError("mlflow is required for the trainer worker: pip install -e '.[training]'") from None

DEFAULT_TRACKING_URI = "http://localhost:5052"
EXPERIMENT_PREFIX = os.environ.get("MAGNIDATA_MLFLOW_EXPERIMENT_PREFIX", "magnidata")
HEARTBEAT_FILENAME = ".trainer_heartbeat"
LOG_DIRNAME = ".trainer_logs"
POLL_S = 5.0
LOG_UPLOAD_S = 30.0
TERM_GRACE_S = 30.0
ERROR_TAG = "magnidata.error"
TERMINAL = {s.value for s in (contract.RunStatus.FINISHED, contract.RunStatus.FAILED, contract.RunStatus.KILLED)}

_PROCESS_TOKEN = uuid.uuid4().hex[:8]
stop_requested = threading.Event()


def worker_id() -> str:
    """Return this process's id, ``host:pid:token``; the token tells restarts with a reused pid apart."""
    return f"{socket.gethostname()}:{os.getpid()}:{_PROCESS_TOKEN}"


def touch_heartbeat(data_root: Path) -> None:
    """Refresh ``data_user/.trainer_heartbeat`` so the API knows a worker is alive."""
    path = data_root / "data_user" / HEARTBEAT_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _experiment_ids(client: Any) -> list[str]:
    prefix = EXPERIMENT_PREFIX + "/"
    return [e.experiment_id for e in client.search_experiments(max_results=1000) if e.name.startswith(prefix)]


def _search(client: Any, status: str, order: str = "ASC") -> list[Any]:
    ids = _experiment_ids(client)
    if not ids:
        return []
    return list(
        client.search_runs(
            ids,
            f"attributes.status = '{status}' and tags.`{contract.TAG_ORIGIN}` = '{contract.ORIGIN_UI}'",
            order_by=[f"attributes.start_time {order}"],
            max_results=100,
        )
    )


def find_scheduled(client: Any) -> list[Any]:
    """Return ``SCHEDULED`` runs launched from the UI, oldest first."""
    return _search(client, contract.RunStatus.SCHEDULED.value)


def find_inference_requests(client: Any) -> list[Any]:
    """Return runs with queued on-demand inference, oldest first."""
    ids = _experiment_ids(client)
    if not ids:
        return []
    return list(
        client.search_runs(
            ids,
            f"tags.`{contract.TAG_INFERENCE_STATUS}` = 'queued'",
            order_by=["attributes.start_time ASC"],
            max_results=100,
        )
    )


def claim(client: Any, run_id: str, me: str | None = None) -> bool:
    """Try to take a run: tag it with this worker's id and confirm the tag stuck.

    Returns False if another worker already tagged it or it is no longer ``SCHEDULED``.
    """
    me = me or worker_id()
    run = client.get_run(run_id)
    if run.info.status != contract.RunStatus.SCHEDULED.value or run.data.tags.get(contract.TAG_WORKER):
        return False
    client.set_tag(run_id, contract.TAG_WORKER, me)
    again = client.get_run(run_id)
    return again.data.tags.get(contract.TAG_WORKER) == me and again.info.status == contract.RunStatus.SCHEDULED.value


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def recover_stale(client: Any, me: str | None = None) -> list[str]:
    """Fail ``RUNNING`` runs that an earlier worker process on this host left behind.

    A run is stale when its worker tag names this host with a dead pid, or with this
    process's pid but a different token (a restart that reused the pid). Runs of other
    hosts and of this very process are left alone. Returns the failed run ids.
    """
    me = me or worker_id()
    host, my_pid, _ = me.split(":")
    failed = []
    for run in _search(client, contract.RunStatus.RUNNING.value):
        tag = run.data.tags.get(contract.TAG_WORKER, "")
        parts = tag.split(":")
        if len(parts) != 3 or parts[0] != host or tag == me or not parts[1].isdigit():
            continue
        if parts[1] == my_pid or not _pid_alive(int(parts[1])):
            client.set_tag(run.info.run_id, ERROR_TAG, "worker restarted during run")
            client.set_terminated(run.info.run_id, contract.RunStatus.FAILED.value)
            failed.append(run.info.run_id)
    return failed


def _last_line(path: Path) -> str:
    try:
        lines = [ln.strip() for ln in path.read_text(errors="replace").splitlines() if ln.strip()]
    except OSError:
        return ""
    return lines[-1][:200] if lines else ""


def _upload_log(client: Any, run_id: str, log: Path) -> None:
    if log.is_file():
        client.log_artifact(run_id, str(log), artifact_path="logs")


def trainer_argv(run_id: str, data_root: Path, tracking_uri: str) -> list[str]:
    """Return the command that trains one run, on the device named by ``MAGNIDATA_TRAIN_DEVICE``."""
    return [
        sys.executable,
        "-m",
        "precisionai.magnidata.training.train",
        "--run-id",
        run_id,
        "--data-root",
        str(data_root),
        "--tracking-uri",
        tracking_uri,
        "--device",
        os.environ.get("MAGNIDATA_TRAIN_DEVICE", "auto"),
    ]


def inference_argv(run_id: str, image_name: str, data_root: Path, tracking_uri: str) -> list[str]:
    """Return the command that generates one persisted prediction mask."""
    return [
        sys.executable,
        "-m",
        "precisionai.magnidata.training.infer",
        "--run-id",
        run_id,
        "--image",
        image_name,
        "--data-root",
        str(data_root),
        "--tracking-uri",
        tracking_uri,
        "--device",
        os.environ.get("MAGNIDATA_TRAIN_DEVICE", "auto"),
    ]


def _stop_child(child: subprocess.Popen) -> None:
    child.terminate()
    try:
        child.wait(TERM_GRACE_S)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()


def _watch(
    child: subprocess.Popen, client: Any, run_id: str, log: Path, beat: Callable[[], None], upload_every: float
) -> bool:
    """Wait for the child while keeping the heartbeat and log fresh; True if it was stopped by a signal."""
    last_upload = time.monotonic()
    while child.poll() is None:
        beat()
        if stop_requested.is_set():
            _stop_child(child)
            return True
        if time.monotonic() - last_upload >= upload_every:
            _upload_log(client, run_id, log)
            last_upload = time.monotonic()
        time.sleep(0.5)
    return False


def run_one(
    client: Any,
    run_id: str,
    data_root: Path,
    argv: Sequence[str] | None = None,
    me: str | None = None,
    log_upload_s: float = LOG_UPLOAD_S,
) -> str | None:
    """Claim and train one run in a subprocess; return its final status, or None if not claimed.

    A trainer that exits without leaving the run in a terminal status fails the run, with
    the exit code and last log line in the ``magnidata.error`` tag.
    """
    if not claim(client, run_id, me):
        return None
    log = data_root / "data_user" / LOG_DIRNAME / run_id / "trainer.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = list(argv) if argv else trainer_argv(run_id, data_root, client.tracking_uri)
    with open(log, "w") as out:
        child = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT)
        interrupted = _watch(child, client, run_id, log, lambda: touch_heartbeat(data_root), log_upload_s)
    status = client.get_run(run_id).info.status
    if status not in TERMINAL:
        if interrupted:
            client.set_terminated(run_id, contract.RunStatus.KILLED.value)
        else:
            detail = f"trainer exited with code {child.returncode}: {_last_line(log)}"
            client.set_tag(run_id, ERROR_TAG, detail[:250])
            client.set_terminated(run_id, contract.RunStatus.FAILED.value)
        status = client.get_run(run_id).info.status
    _upload_log(client, run_id, log)
    return status


def run_inference(client: Any, run_id: str, data_root: Path, argv: Sequence[str] | None = None) -> bool:
    """Process one queued inference request and persist its completion status."""
    run = client.get_run(run_id)
    if run.data.tags.get(contract.TAG_INFERENCE_STATUS) != "queued":
        return False
    image_name = run.data.tags.get(contract.TAG_INFERENCE_IMAGE, "")
    client.set_tag(run_id, contract.TAG_INFERENCE_STATUS, "running")
    log = data_root / "data_user" / LOG_DIRNAME / run_id / "inference.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = list(argv) if argv else inference_argv(run_id, image_name, data_root, client.tracking_uri)
    with open(log, "w") as out:
        child = subprocess.Popen(command, stdout=out, stderr=subprocess.STDOUT)
        interrupted = _watch(child, client, run_id, log, lambda: touch_heartbeat(data_root), LOG_UPLOAD_S)
    _upload_log(client, run_id, log)
    if child.returncode == 0 and not interrupted:
        client.set_tag(run_id, contract.TAG_INFERENCE_ERROR, "")
        client.set_tag(run_id, contract.TAG_INFERENCE_STATUS, "ready")
    else:
        detail = (
            "inference stopped with the worker"
            if interrupted
            else f"inference exited with code {child.returncode}: {_last_line(log)}"
        )
        client.set_tag(run_id, contract.TAG_INFERENCE_ERROR, detail[:250])
        client.set_tag(run_id, contract.TAG_INFERENCE_STATUS, "error")
    return True


def loop(client: Any, data_root: Path, max_iterations: int | None = None, poll_s: float = POLL_S) -> None:
    """Poll for scheduled runs until stopped (or for ``max_iterations`` passes)."""
    iteration = 0
    while not stop_requested.is_set() and (max_iterations is None or iteration < max_iterations):
        iteration += 1
        touch_heartbeat(data_root)
        try:
            handled = False
            for run in find_scheduled(client):
                if run_one(client, run.info.run_id, data_root) is not None:
                    handled = True
                    break
            if not handled:
                for run in find_inference_requests(client):
                    if run_inference(client, run.info.run_id, data_root):
                        break
        except Exception as err:
            print(f"worker: {type(err).__name__}: {err}", file=sys.stderr, flush=True)
        if max_iterations is None or iteration < max_iterations:
            stop_requested.wait(poll_s)


def main(argv: Sequence[str] | None = None) -> int:
    """Run the worker until SIGTERM or SIGINT."""
    parser = argparse.ArgumentParser(description="Run scheduled MagniData training runs from MLflow.")
    parser.add_argument("--data-root", type=Path, default=Path(os.environ.get("DATA_ROOT", ".")))
    parser.add_argument("--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    parser.add_argument("--poll-s", type=float, default=POLL_S)
    args = parser.parse_args(argv)

    signal.signal(signal.SIGTERM, lambda *_: stop_requested.set())
    signal.signal(signal.SIGINT, lambda *_: stop_requested.set())
    client = mlflow.MlflowClient(tracking_uri=args.tracking_uri)
    print(f"worker {worker_id()} watching {args.tracking_uri}", flush=True)
    touch_heartbeat(args.data_root)
    for run_id in recover_stale(client):
        print(f"worker: marked stale run {run_id} as FAILED", flush=True)
    loop(client, args.data_root, poll_s=args.poll_s)
    return 0


if __name__ == "__main__":
    sys.exit(main())
