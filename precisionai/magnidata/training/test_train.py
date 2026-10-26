# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests of the reference trainer on a synthetic dataset, on CPU with tiny_unet."""

import csv

import numpy as np
import pytest
from PIL import Image

from precisionai.magnidata.training import contract

pytest.importorskip("torch")
train = pytest.importorskip("precisionai.magnidata.training.train")
models = pytest.importorskip("precisionai.magnidata.training.models")

QUICK = {
    "epochs": "2",
    "batch_size": "4",
    "image_size": "32",
    "lr": "0.01",
    "score_every": "1",
    "top_k_predictions": "3",
    "num_workers": "0",
}


def _make_learnable(ds):
    """Make each image a noisy copy of its mask so the task is trivially learnable."""
    rng = np.random.default_rng(1)
    for name in ds.names:
        mask = np.asarray(Image.open(ds.root / "data_user" / "syn" / "masks" / name).convert("RGB"))
        noise = rng.integers(0, 20, mask.shape, dtype=np.uint8)
        Image.fromarray(np.clip(mask.astype(int) + noise, 0, 255).astype(np.uint8)).save(
            ds.root / "data_user" / "syn" / "images" / name
        )


def _new_run(make_run, params=None, tags=None, source="data_user/syn.csv"):
    uri, client, run_id = make_run()
    for key, value in {**QUICK, **(params or {})}.items():
        client.log_param(run_id, key, value)
    for key, value in {contract.TAG_DATASET_SOURCE: source, **(tags or {})}.items():
        client.set_tag(run_id, key, value)
    return uri, client, run_id


def _artifacts(client, run_id, path):
    return sorted(f.path for f in client.list_artifacts(run_id, path) if not f.is_dir)


def _history(client, run_id, key):
    return [m.value for m in client.get_metric_history(run_id, key)]


def test_two_epoch_run_logs_everything(mlflow_tracking, synthetic_dataset):
    uri, client, run_id = _new_run(mlflow_tracking)
    assert train.run(run_id, uri, synthetic_dataset.root, "cpu") == 0

    run = client.get_run(run_id)
    assert run.info.status == "FINISHED"
    assert len(_history(client, run_id, "train_loss/loss")) == 2
    assert len(_history(client, run_id, "val_loss/dice")) == 2
    assert _history(client, run_id, "val_metric/miou")
    assert _history(client, run_id, "val_metric/iou_background")
    assert _history(client, run_id, "Learning_rate/lr")
    assert run.data.tags[contract.TAG_SCORED_EPOCHS] == "1,2"
    assert run.data.tags[contract.TAG_PER_SAMPLE_SCHEMA] == contract.SCHEMA_VERSION
    assert run.data.params["num_classes"] == "3"

    assert _artifacts(client, run_id, "magnidata/per_sample") == [
        contract.per_sample_path(1),
        contract.per_sample_path(2),
    ]
    assert contract.SPLIT_ARTIFACT in _artifacts(client, run_id, "magnidata")
    assert contract.PALETTE_ARTIFACT in _artifacts(client, run_id, "magnidata")
    assert _artifacts(client, run_id, "checkpoints") == [contract.CHECKPOINT_ARTIFACT]
    predictions = _artifacts(client, run_id, "magnidata/predictions/epoch_0002")
    assert 0 < len(predictions) <= 3

    path = client.download_artifacts(run_id, contract.per_sample_path(2))
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert sorted(r["image_name"] for r in rows) == synthetic_dataset.names
    assert set(rows[0]) == {"image_name", "split", "loss_total", "loss_dice", "loss_cce", "confidence", "iou"}
    contract.validate_header(list(rows[0]), ["dice", "cce"])


def test_run_reports_device_and_progress(mlflow_tracking, synthetic_dataset, capsys):
    uri, client, run_id = _new_run(mlflow_tracking)
    assert train.run(run_id, uri, synthetic_dataset.root, "cpu") == 0

    out = capsys.readouterr().out
    assert "device=cpu" in out
    assert "epoch 1/2 train batch" in out
    assert "epoch 1/2 done" in out
    assert "epoch 2/2 done" in out
    assert "scored" in out
    run = client.get_run(run_id)
    assert run.data.tags[contract.TAG_DEVICE] == "cpu"
    assert run.data.tags[contract.TAG_ETA_SECONDS] == "0"
    assert run.data.tags[contract.TAG_PROGRESS_STAGE] == "Complete"
    assert run.data.tags[contract.TAG_PROGRESS_CURRENT] == "1"
    assert run.data.tags[contract.TAG_PROGRESS_TOTAL] == "1"
    assert len(_history(client, run_id, contract.EPOCH_SECONDS_METRIC)) == 2
    assert all(v > 0 for v in _history(client, run_id, contract.SAMPLES_PER_SEC_METRIC))


def test_run_with_loader_workers_matches_columns(mlflow_tracking, synthetic_dataset):
    uri, client, run_id = _new_run(mlflow_tracking, {"num_workers": "2", "epochs": "1"})
    assert train.run(run_id, uri, synthetic_dataset.root, "cpu") == 0
    assert client.get_run(run_id).info.status == "FINISHED"
    path = client.download_artifacts(run_id, contract.per_sample_path(1))
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert set(rows[0]) == {"image_name", "split", "loss_total", "loss_dice", "loss_cce", "confidence", "iou"}
    assert (synthetic_dataset.root / "data_user" / ".train_cache").is_dir()


def test_amp_is_off_on_cpu_and_params_logged(mlflow_tracking, synthetic_dataset):
    assert train.make_amp(train.torch.device("cpu"), True) is None
    uri, client, run_id = _new_run(mlflow_tracking, {"amp": "true"})
    assert train.run(run_id, uri, synthetic_dataset.root, "cpu") == 0
    assert client.get_run(run_id).data.params["amp"] == "true"


@pytest.mark.skipif(not train.torch.cuda.is_available(), reason="needs CUDA")
def test_amp_runs_on_cuda(mlflow_tracking, synthetic_dataset):
    uri, _, run_id = _new_run(mlflow_tracking, {"amp": "true", "epochs": "1"})
    assert train.run(run_id, uri, synthetic_dataset.root, "cuda") == 0


@pytest.mark.skipif(train.torch.cuda.is_available(), reason="CUDA is available")
def test_resolve_device_cuda_unavailable_raises():
    with pytest.raises(RuntimeError, match="CUDA requested"):
        train.resolve_device("cuda")


@pytest.mark.skipif(train.torch.cuda.is_available(), reason="CUDA is available")
def test_resolve_device_auto_falls_back_to_cpu_with_warning(capsys):
    assert train.resolve_device("auto").type == "cpu"
    assert "CUDA not available" in capsys.readouterr().out


@pytest.mark.skipif(train.torch.cuda.is_available(), reason="CUDA is available")
def test_cuda_request_without_cuda_fails_run(mlflow_tracking, synthetic_dataset):
    uri, client, run_id = _new_run(mlflow_tracking)
    assert train.run(run_id, uri, synthetic_dataset.root, "cuda") == 1
    run = client.get_run(run_id)
    assert run.info.status == "FAILED"
    assert run.data.tags["magnidata.error"].startswith("CUDA requested but not available")


def test_epoch_one_and_last_are_always_scored(mlflow_tracking, synthetic_dataset):
    uri, client, run_id = _new_run(mlflow_tracking, {"epochs": "4", "score_every": "3"})
    assert train.run(run_id, uri, synthetic_dataset.root, "cpu") == 0
    run = client.get_run(run_id)
    assert run.data.tags[contract.TAG_SCORED_EPOCHS] == "1,3,4"
    assert run.data.tags[contract.TAG_HAS_SPLIT] == "true"


def test_split_is_tagged_before_the_first_epoch(mlflow_tracking, synthetic_dataset, tmp_path):
    uri, client, run_id = _new_run(mlflow_tracking)
    cfg = train.config_from_run(client.get_run(run_id).data.params, client.get_run(run_id).data.tags)
    cfg.num_workers = 0
    sink = train.MlflowSink(run_id, uri)
    recorder = train.LossRecorder(cfg.loss_terms, sink, tmp_path / "scratch")
    train.prepare_data(cfg, synthetic_dataset.root, recorder)
    tags = client.get_run(run_id).data.tags
    assert tags[contract.TAG_HAS_SPLIT] == "true"
    assert contract.TAG_SCORED_EPOCHS not in tags
    assert contract.SPLIT_ARTIFACT in _artifacts(client, run_id, "magnidata")


def test_loss_decreases_on_learnable_data(mlflow_tracking, synthetic_dataset):
    _make_learnable(synthetic_dataset)
    uri, client, run_id = _new_run(mlflow_tracking, {"batch_size": "2", "epochs": "4", "lr": "0.02"})
    assert train.run(run_id, uri, synthetic_dataset.root, "cpu") == 0

    def mean_total(epoch):
        path = client.download_artifacts(run_id, contract.per_sample_path(epoch))
        with open(path, newline="") as fh:
            return np.mean([float(r["loss_total"]) for r in csv.DictReader(fh)])

    assert mean_total(4) < mean_total(1)


def test_cancel_before_start_kills_run(mlflow_tracking, synthetic_dataset):
    uri, client, run_id = _new_run(mlflow_tracking, tags={contract.TAG_CANCEL_REQUESTED: "true"})
    assert train.run(run_id, uri, synthetic_dataset.root, "cpu") == 0
    assert client.get_run(run_id).info.status == "KILLED"
    assert _history(client, run_id, "train_loss/loss") == []


def test_missing_dataset_fails_run(mlflow_tracking, synthetic_dataset, capsys):
    uri, client, run_id = _new_run(mlflow_tracking, source="data_user/nope.csv")
    assert train.run(run_id, uri, synthetic_dataset.root, "cpu") != 0
    run = client.get_run(run_id)
    assert run.info.status == "FAILED"
    assert "dataset CSV not found" in run.data.tags["magnidata.error"]
    assert "Traceback" in capsys.readouterr().out


def test_unknown_preset_fails_run(mlflow_tracking, synthetic_dataset):
    uri, client, run_id = _new_run(mlflow_tracking, tags={contract.TAG_PRESET: "giant"})
    assert train.run(run_id, uri, synthetic_dataset.root, "cpu") != 0
    assert "unknown preset" in client.get_run(run_id).data.tags["magnidata.error"]


def test_cli_without_run_id_creates_external_run(mlflow_tracking, synthetic_dataset, tmp_path):
    uri, client, _ = mlflow_tracking()
    code = train.main(
        [
            "--dataset", "data_user/syn.csv",
            "--data-root", str(synthetic_dataset.root),
            "--tracking-uri", uri,
            "--experiment", "cli",
            "--artifact-location", (tmp_path / "cli-artifacts").as_uri(),
            "--device", "cpu",
            *[arg for k, v in QUICK.items() for arg in ("--param", f"{k}={v}")],
        ]
    )  # fmt: skip
    assert code == 0
    experiment = client.get_experiment_by_name("cli")
    assert experiment is not None
    (run,) = client.search_runs([experiment.experiment_id])
    assert run.info.status == "FINISHED"
    assert run.data.tags[contract.TAG_ORIGIN] == contract.ORIGIN_EXTERNAL
    assert run.data.tags[contract.TAG_DATASET_SOURCE] == "data_user/syn.csv"


def test_cli_requires_run_or_dataset():
    with pytest.raises(SystemExit):
        train.main([])


def test_config_defaults_and_overrides():
    cfg = train.config_from_run(
        {"epochs": "3", "weight_dice": "2.5", "pretrained": "false"},
        {contract.TAG_DATASET_SOURCE: "x.csv", contract.TAG_LOSS_TERMS: "dice,bce"},
    )
    assert (cfg.epochs, cfg.loss_terms, cfg.loss_weights, cfg.pretrained) == (3, ["dice", "bce"], {"dice": 2.5}, False)
    assert cfg.as_params()["weight_dice"] == "2.5"
    with pytest.raises(ValueError, match="dataset_source"):
        train.config_from_run({}, {})


def test_iou_from_confusion():
    torch = pytest.importorskip("torch")
    miou, per_class = train.iou_from_confusion(torch.tensor([[4, 0, 0], [1, 1, 0], [0, 0, 0]]))
    assert per_class == pytest.approx([4 / 5, 1 / 2, None])
    assert miou == pytest.approx((4 / 5 + 1 / 2) / 2)


@pytest.mark.parametrize("preset", contract.MODEL_PRESETS)
def test_models_output_logits_at_input_resolution(preset):
    torch = pytest.importorskip("torch")
    model = models.build_model(preset, num_classes=4, pretrained=False).eval()
    with torch.no_grad():
        assert model(torch.zeros(2, 3, 40, 56)).shape == (2, 4, 40, 56)


def test_unknown_model_preset():
    with pytest.raises(ValueError, match="unknown preset"):
        models.build_model("giant", 3)


def _add_split_column(ds, labels):
    with open(ds.csv, newline="") as fh:
        rows = list(csv.DictReader(fh))
    with open(ds.csv, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[*rows[0], "split"])
        writer.writeheader()
        for row, label in zip(rows, labels, strict=True):
            writer.writerow({**row, "split": label})


def test_column_split_mode_uses_csv_labels(mlflow_tracking, synthetic_dataset):
    labels = ["train"] * 5 + ["val"] * 2 + ["test"]
    _add_split_column(synthetic_dataset, labels)
    uri, client, run_id = _new_run(mlflow_tracking, {"split_mode": "column", "epochs": "1"})
    assert train.run(run_id, uri, synthetic_dataset.root, "cpu") == 0
    path = client.download_artifacts(run_id, contract.SPLIT_ARTIFACT)
    with open(path, newline="") as fh:
        assert [r["split"] for r in csv.DictReader(fh)] == labels


def test_column_split_mode_without_column_fails(mlflow_tracking, synthetic_dataset):
    uri, client, run_id = _new_run(mlflow_tracking, {"split_mode": "column"})
    assert train.run(run_id, uri, synthetic_dataset.root, "cpu") != 0
    assert "no split column" in client.get_run(run_id).data.tags["magnidata.error"]


def test_random_split_mode_ignores_csv_column(mlflow_tracking, synthetic_dataset):
    _add_split_column(synthetic_dataset, ["test"] * 8)
    uri, client, run_id = _new_run(mlflow_tracking, {"epochs": "1"})
    assert train.run(run_id, uri, synthetic_dataset.root, "cpu") == 0
    path = client.download_artifacts(run_id, contract.SPLIT_ARTIFACT)
    with open(path, newline="") as fh:
        assert {r["split"] for r in csv.DictReader(fh)} == {"train", "val"}


def test_unknown_split_mode():
    with pytest.raises(ValueError, match="split_mode"):
        train.config_from_run({"split_mode": "stratified"}, {contract.TAG_DATASET_SOURCE: "x.csv"})


def test_native_resolution_trains_at_the_images_own_size(mlflow_tracking, synthetic_dataset, tmp_path):
    torch = pytest.importorskip("torch")
    uri, client, run_id = _new_run(
        mlflow_tracking, {"input_resolution": "native", "epochs": "1", "batch_size": "2", "num_workers": "0"}
    )

    assert train.run(run_id, uri, synthetic_dataset.root, "cpu") == 0

    height, width = synthetic_dataset.size
    checkpoint = torch.load(
        client.download_artifacts(run_id, contract.CHECKPOINT_ARTIFACT, str(tmp_path)), weights_only=True
    )
    assert checkpoint["input_size"] == [width, height]
    assert client.get_run(run_id).data.params["input_resolution"] == "native"
    assert (synthetic_dataset.root / "data_user" / ".train_cache" / f"{width}x{height}").is_dir()


def test_native_resolution_resizes_images_of_another_size(mlflow_tracking, synthetic_dataset, tmp_path):
    torch = pytest.importorskip("torch")
    name = synthetic_dataset.names[0]
    folder = synthetic_dataset.root / "data_user" / "syn"
    Image.new("RGB", (40, 30)).save(folder / "images" / name)
    Image.new("RGB", (40, 30)).save(folder / "masks" / name)
    uri, client, run_id = _new_run(
        mlflow_tracking, {"input_resolution": "native", "epochs": "1", "batch_size": "2", "num_workers": "0"}
    )

    assert train.run(run_id, uri, synthetic_dataset.root, "cpu") == 0

    height, width = synthetic_dataset.size
    checkpoint = torch.load(
        client.download_artifacts(run_id, contract.CHECKPOINT_ARTIFACT, str(tmp_path)), weights_only=True
    )
    assert checkpoint["input_size"] == [width, height]


def test_unknown_input_resolution_is_rejected():
    with pytest.raises(ValueError, match="unknown input_resolution 'full': choose image_size or native"):
        train.config_from_run({"input_resolution": "full"}, {contract.TAG_DATASET_SOURCE: "data_user/syn.csv"})
