# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Keep openapi.yaml in step with the Flask routes and with what the endpoints really return.

Run:  python -m pytest precisionai/magnidata/api/test_openapi.py -q
"""

import io
import json
import re
from pathlib import Path
from types import ModuleType
from typing import Any

import experiments
import mlflow_store as store_module
import pytest
import yaml
from conftest import load_app, seed_run
from jsonschema import Draft202012Validator
from openapi_spec_validator import validate
from PIL import Image
from referencing import Registry, Resource
from referencing.jsonschema import DRAFT202012
from werkzeug.test import TestResponse

from precisionai.magnidata.training import contract

SPEC_PATH = Path(__file__).with_name("openapi.yaml")
SPEC_URI = "urn:magnidata:openapi"
FLASK_ARGUMENT = re.compile(r"<(?:[^:<>]+:)?([^<>]+)>")
SPEC = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
REGISTRY = Registry().with_resource(SPEC_URI, Resource.from_contents(SPEC, default_specification=DRAFT202012))


def _escape(token: str) -> str:
    return token.replace("~", "~0").replace("/", "~1")


def _at(pointer: str) -> dict:
    node = SPEC
    for raw in pointer.lstrip("/").split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        node = node[int(token)] if isinstance(node, list) else node[token]
    return node


def _follow(pointer: str) -> tuple[dict, str]:
    """Return the node at ``pointer`` and its own pointer, after following any chain of local ``$ref``."""
    node = _at(pointer)
    while "$ref" in node:
        pointer = node["$ref"].removeprefix("#")
        node = _at(pointer)
    return node, pointer


def _schema_errors(schema_pointer: str, instance: Any) -> list[str]:
    validator = Draft202012Validator({"$ref": f"{SPEC_URI}#{schema_pointer}"}, registry=REGISTRY)
    return sorted(
        f"{'/'.join(map(str, e.absolute_path)) or '<root>'}: {e.message}" for e in validator.iter_errors(instance)
    )


def _get_routes(app_module: ModuleType) -> dict[str, set[str]]:
    """Return every GET route under /api/ as an OpenAPI path template, with its path arguments."""
    return {
        FLASK_ARGUMENT.sub(r"{\1}", rule.rule): set(rule.arguments)
        for rule in app_module.app.url_map.iter_rules()
        if rule.rule.startswith("/api/") and "GET" in (rule.methods or ())
    }


def _documented_get_paths() -> set[str]:
    return {path for path, item in SPEC["paths"].items() if "get" in item}


def _path_parameters(path: str) -> set[str]:
    operation = SPEC["paths"][path]["get"]
    params = [
        _follow(f"/paths/{_escape(path)}/get/parameters/{i}")[0] for i in range(len(operation.get("parameters", [])))
    ]
    return {p["name"] for p in params if p["in"] == "path"}


@pytest.fixture
def app_module(tmp_path, monkeypatch):
    return load_app(tmp_path, monkeypatch, MAGNIDATA_DEMO_MODE=None, MAGNIDATA_STATIC_DIR=None)


def test_document_is_valid_openapi_3_1():
    validate(SPEC)
    assert SPEC["openapi"].startswith("3.1.")


def test_every_get_route_is_documented(app_module):
    missing = sorted(set(_get_routes(app_module)) - _documented_get_paths())
    assert not missing, f"GET routes missing from openapi.yaml: {missing}"


def test_every_documented_path_is_a_get_route(app_module):
    stale = sorted(_documented_get_paths() - set(_get_routes(app_module)))
    assert not stale, f"openapi.yaml documents GET paths the app does not serve: {stale}"


def test_path_parameters_match_route_arguments(app_module):
    for path, arguments in _get_routes(app_module).items():
        assert _path_parameters(path) == arguments, path


def test_operation_ids_are_unique():
    ids = [op["operationId"] for item in SPEC["paths"].values() for op in item.values() if isinstance(op, dict)]
    assert len(ids) == len(set(ids))


def test_examples_match_their_schemas():
    checked = 0
    for path, item in SPEC["paths"].items():
        for status in item["get"]["responses"]:
            response, pointer = _follow(f"/paths/{_escape(path)}/get/responses/{status}")
            for media, content in response.get("content", {}).items():
                if media == "application/json" and "example" in content:
                    errors = _schema_errors(f"{pointer}/content/{_escape(media)}/schema", content["example"])
                    assert not errors, f"GET {path} {status} example: {errors}"
                    checked += 1
    assert checked > 20


def test_served_document_is_the_file(app_module):
    resp = app_module.app.test_client().get("/api/openapi.json")
    assert resp.status_code == 200
    assert resp.mimetype == "application/json"
    assert resp.get_json() == SPEC


def test_a_deployment_without_the_document_answers_json_503(app_module, tmp_path, monkeypatch):
    # The Space image is built by its own copy of the Dockerfile, which can leave openapi.yaml out.
    monkeypatch.setattr(app_module, "OPENAPI_PATH", str(tmp_path / "missing" / "openapi.yaml"))
    app_module._openapi_document.cache_clear()
    resp = app_module.app.test_client().get("/api/openapi.json")
    assert resp.status_code == 503
    assert resp.mimetype == "application/json"
    assert "OpenAPI document" in resp.get_json()["error"]
    _assert_documented("/api/openapi.json", resp)


def _assert_documented(template: str, resp: TestResponse) -> None:
    """Fail unless ``resp`` is a status, media type and (for JSON) body that ``template`` documents."""
    assert template in SPEC["paths"], template
    status = str(resp.status_code)
    assert status in SPEC["paths"][template]["get"]["responses"], f"GET {template} answered undocumented {status}"
    response, pointer = _follow(f"/paths/{_escape(template)}/get/responses/{status}")
    mimetype = resp.mimetype or ""
    media_types = response.get("content", {})
    media = next(
        (m for m in media_types if m == mimetype or (m.endswith("/*") and mimetype.startswith(m[:-1]))),
        None,
    )
    assert media is not None, f"GET {template} {status} answered undocumented {mimetype!r}"
    if media == "application/json":
        errors = _schema_errors(f"{pointer}/content/{_escape(media)}/schema", json.loads(resp.get_data()))
        assert not errors, f"GET {template} {status}: {errors}"


def _png_bytes(color: tuple[int, int, int], size: tuple[int, int] = (8, 6)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def served(tmp_path, monkeypatch, synthetic_dataset, mlflow_store):
    """An app with one dataset (embeddings, a comparison variant, a note, a partition) and one scored run."""
    app_module = load_app(tmp_path, monkeypatch, MAGNIDATA_DEMO_MODE=None, MAGNIDATA_STATIC_DIR=None)
    monkeypatch.chdir(tmp_path)  # image endpoints open paths relative to the working directory (/app in Docker)
    (tmp_path / "confi.yaml").write_text(
        "datasets:\n  - name: Synthetic\n    description: test set\n    source: data_user/syn.csv\n"
    )
    vectors = {name: [float(i), float(i % 3), float(i % 2), 1.0] for i, name in enumerate(synthetic_dataset.names)}
    (tmp_path / "data_user" / "syn.json").write_text(json.dumps({"embeddings": vectors}))
    (tmp_path / "data_user" / "syn_alt.json").write_text(json.dumps({"embeddings": vectors}))
    (tmp_path / "data_user" / "syn" / "images" / "bare.png").write_bytes(_png_bytes((1, 2, 3)))
    app_module.db.create_note("img_00", "data_user/syn/images/img_00.png", "looks fine")
    app_module.db.commit_partitions("data_user/syn.csv", [{"image_name": "img_00.png", "cluster_id": 1}])

    store_module.set_store(mlflow_store)
    experiments.clear_cache()
    per_sample = {
        contract.per_sample_path(epoch): (
            f"image_name,split,loss_total,loss_dice,iou\nimg_00.png,train,{a},{a / 2},0.5\nimg_01.png,val,{b},{b / 2},0.6\n"
        )
        for epoch, (a, b) in {1: (0.9, 0.2), 2: (0.7, 0.3), 3: (0.5, 0.4)}.items()
    }
    tags = {
        contract.TAG_PER_SAMPLE_SCHEMA: contract.SCHEMA_VERSION,
        contract.TAG_SCORED_EPOCHS: "1,2,3",
        contract.TAG_DATASET_SOURCE: "data_user/syn.csv",
    }
    run_id = seed_run(mlflow_store, metrics={"val_loss/loss": [0.9, 0.5]}, tags=tags, artifacts=per_sample)
    prediction = tmp_path / "prediction" / "img_00.png"
    prediction.parent.mkdir()
    prediction.write_bytes(_png_bytes((255, 0, 0)))
    mlflow_store._client.log_artifact(run_id, str(prediction), "magnidata/predictions/epoch_0003")
    yield app_module.app.test_client(), run_id
    experiments.clear_cache()


SYN = "source=data_user/syn.csv"
IMAGE = "data_user/syn/images/img_00.png"
MASK = "data_user/syn/masks/img_00.png"


@pytest.mark.parametrize(
    ("template", "url", "status"),
    [
        ("/api/health", "/api/health", 200),
        ("/api/config", "/api/config", 200),
        ("/api/mlflow/status", "/api/mlflow/status", 200),
        ("/api/datasets", "/api/datasets", 200),
        ("/api/datasets/summary", f"/api/datasets/summary?{SYN}", 200),
        ("/api/datasets/summary", "/api/datasets/summary", 400),
        ("/api/datasets/summary", "/api/datasets/summary?source=data_user/missing.csv", 404),
        ("/api/dataset/csv", f"/api/dataset/csv?{SYN}", 200),
        ("/api/dataset/csv", "/api/dataset/csv?source=data_user/missing.csv", 404),
        ("/api/dataset/embeddings", f"/api/dataset/embeddings?{SYN}", 200),
        ("/api/dataset/embeddings", f"/api/dataset/embeddings?{SYN}&variant=missing", 404),
        ("/api/dataset/embeddings/variants", f"/api/dataset/embeddings/variants?{SYN}", 200),
        ("/api/embedding/projection", f"/api/embedding/projection?{SYN}&method=pca", 200),
        ("/api/embedding/projection", f"/api/embedding/projection?{SYN}&method=bogus", 400),
        ("/api/embedding/clusters", f"/api/embedding/clusters?{SYN}&k=3", 200),
        ("/api/embedding/compare", f"/api/embedding/compare?{SYN}&variant=alt", 200),
        ("/api/embedding/compare", f"/api/embedding/compare?{SYN}", 400),
        ("/api/embedding/analysis/{job_id}", "/api/embedding/analysis/unknown", 404),
        ("/api/image", f"/api/image?path={IMAGE}&max_size=64", 200),
        ("/api/image", "/api/image", 400),
        ("/api/image", "/api/image?path=data_user/missing.png", 500),
        ("/api/mask", f"/api/mask?path={MASK}", 200),
        ("/api/overlay", f"/api/overlay?img={IMAGE}&mask={MASK}", 200),
        ("/api/mask/debug", f"/api/mask/debug?path={MASK}", 200),
        ("/api/annotation/mask", f"/api/annotation/mask?path={IMAGE}", 200),
        ("/api/annotation/mask", "/api/annotation/mask?path=data_user/syn/images/bare.png", 404),
        ("/api/annotation/overlay", f"/api/annotation/overlay?path={IMAGE}&alpha=0.3", 200),
        ("/api/annotation/overlay", "/api/annotation/overlay?path=/etc/hosts", 400),
        ("/api/notes", "/api/notes", 200),
        ("/api/partitions", f"/api/partitions?{SYN}", 200),
        ("/api/partitions", "/api/partitions", 400),
        ("/api/datasets/demos", "/api/datasets/demos", 200),
        ("/api/datasets/build/limits", "/api/datasets/build/limits", 200),
        ("/api/datasets/build/local-folders", "/api/datasets/build/local-folders", 200),
        ("/api/datasets/build/{job_id}", "/api/datasets/build/unknown", 404),
        ("/api/datasets/experiment-counts", "/api/datasets/experiment-counts", 200),
        ("/api/experiments", "/api/experiments", 200),
        ("/api/runs", "/api/runs", 200),
        ("/api/runs", "/api/runs?per_sample=bogus", 400),
        ("/api/runs/{run_id}", "/api/runs/{run}", 200),
        ("/api/runs/{run_id}", "/api/runs/not-a-run-id", 400),
        ("/api/runs/{run_id}", f"/api/runs/{'0' * 32}", 404),
        ("/api/runs/{run_id}/metrics", "/api/runs/{run}/metrics", 200),
        ("/api/runs/{run_id}/artifacts", "/api/runs/{run}/artifacts?path=magnidata", 200),
        (
            "/api/runs/{run_id}/artifact",
            "/api/runs/{run}/artifact?path=magnidata/predictions/epoch_0003/img_00.png",
            200,
        ),
        ("/api/runs/{run_id}/columns", "/api/runs/{run}/columns", 200),
        ("/api/runs/{run_id}/match", f"/api/runs/{{run}}/match?dataset={SYN.removeprefix('source=')}", 200),
        ("/api/runs/{run_id}/samples/{image_name}", "/api/runs/{run}/samples/img_00.png", 200),
        ("/api/runs/{run_id}/inference", "/api/runs/{run}/inference?image=img_00.png", 200),
        ("/api/runs/{run_id}/inference", "/api/runs/{run}/inference?image=img_01.png", 200),
        ("/api/runs/{run_id}/prediction", "/api/runs/{run}/prediction?image=img_00.png&max_size=64", 200),
        ("/api/runs/compare-samples", "/api/runs/compare-samples?a={run}&b={run}", 200),
        ("/api/training/presets", "/api/training/presets", 200),
    ],
)
def test_responses_match_the_document(served, template, url, status):
    client, run_id = served
    resp = client.get(url.replace("{run}", run_id))
    assert resp.status_code == status
    _assert_documented(template, resp)


@pytest.mark.parametrize(
    ("template", "url", "status"),
    [
        ("/api/config", "/api/config", 200),
        ("/api/mlflow/status", "/api/mlflow/status", 200),
        ("/api/mask/debug", "/api/mask/debug?path=x.png", 403),
        ("/api/image", "/api/image?path=/etc/hosts", 400),
        ("/api/overlay", "/api/overlay?img=https://example.com/a.png&mask=m.png", 400),
        ("/api/experiments", "/api/experiments", 503),
        ("/api/runs/{run_id}", f"/api/runs/{'0' * 32}", 503),
        ("/api/datasets/experiment-counts", "/api/datasets/experiment-counts", 200),
    ],
)
def test_demo_mode_responses_match_the_document(tmp_path, monkeypatch, template, url, status):
    app_module = load_app(tmp_path, monkeypatch, MAGNIDATA_DEMO_MODE="1", MAGNIDATA_STATIC_DIR=None)
    experiments.clear_cache()
    resp = app_module.app.test_client().get(url)
    assert resp.status_code == status
    _assert_documented(template, resp)
