# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

"""Tests for the dataset partitioning command-line tool."""

import csv
import json
import re
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import ClassVar

import pytest
from PIL import Image

from precisionai.magnidata.scripts import partition_dataset as tool


def test_tile_origins_cover_full_tiles_only() -> None:
    assert tool.tile_origins(512, 256, 256) == [(0, 0), (256, 0)]
    assert tool.tile_origins(600, 300, 256) == [(0, 0), (256, 0)]
    assert tool.tile_origins(100, 100, 256) == []


def test_tile_name_is_zero_padded() -> None:
    assert tool.tile_name("pai-a", 256, 0) == "pai-a_x00256_y00000.png"


def test_split_images_balances_tiles_and_never_shares_an_image() -> None:
    counts = {"a": 10, "b": 9, "c": 4, "d": 4, "e": 3}

    groups = tool.split_images(counts, 2, seed=1)

    assert sorted(name for group in groups for name in group) == sorted(counts)
    totals = [sum(counts[name] for name in group) for group in groups]
    assert abs(totals[0] - totals[1]) <= min(counts.values())


def test_split_images_is_deterministic_per_seed() -> None:
    counts = {name: 5 for name in "abcdef"}

    assert tool.split_images(counts, 3, seed=7) == tool.split_images(counts, 3, seed=7)


def test_split_images_rejects_zero_parts() -> None:
    with pytest.raises(ValueError, match="parts must be at least 1"):
        tool.split_images({"a": 1}, 0, seed=0)


def test_split_tiles_sizes_differ_by_at_most_one() -> None:
    tiles = [("a", x, 0) for x in range(7)]

    groups = tool.split_tiles(tiles, 3, seed=0)

    assert sorted(len(group) for group in groups) == [2, 2, 3]
    assert sorted(tile for group in groups for tile in group) == sorted(tiles)


def test_part_folder_slugifies_the_prefix() -> None:
    assert tool.part_folder("AgriStress-500 tiles", 2) == "agristress-500_tiles-part2"


@pytest.fixture
def data_root(tmp_path: Path) -> Path:
    """A data root with a 3-image source dataset; two images have masks, one is smaller than a tile."""
    root = tmp_path / "root"
    images = root / "data_user" / "src" / "images"
    masks = root / "data_user" / "src" / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    for name, size in (("a", (512, 256)), ("b", (256, 512)), ("tiny", (100, 100))):
        Image.new("RGB", size, (10, 120, 10)).save(images / f"{name}.png")
    for name, size in (("a", (512, 256)), ("b", (256, 512))):
        Image.new("RGB", size, (0, 0, 255)).save(masks / f"{name}.png")
    with open(root / "data_user" / "src.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["image_path", "width", "height"])
        for name in ("a", "b", "tiny"):
            writer.writerow([f"data_user/src/images/{name}.png", 0, 0])
    return root


def test_read_source_counts_tiles_and_finds_masks(data_root: Path) -> None:
    images = tool.read_source(data_root / "data_user" / "src.csv", data_root, 256, None, 0)

    by_name = {image.name: image for image in images}
    assert by_name["a.png"].tiles == 2
    assert by_name["b.png"].tiles == 2
    assert "tiny.png" not in by_name  # smaller than one tile
    assert by_name["a.png"].mask is not None


def test_read_source_limit_is_a_seeded_sample(data_root: Path) -> None:
    first = tool.read_source(data_root / "data_user" / "src.csv", data_root, 256, 2, 5)
    second = tool.read_source(data_root / "data_user" / "src.csv", data_root, 256, 2, 5)

    assert [image.name for image in first] == [image.name for image in second]
    assert 1 <= len(first) <= 2  # the sample may include the image too small to tile


def test_write_part_crops_images_and_masks_and_writes_a_manifest(data_root: Path, tmp_path: Path) -> None:
    images = tool.read_source(data_root / "data_user" / "src.csv", data_root, 256, None, 0)
    by_name = {image.name: image for image in images}
    folder = tmp_path / "image_sets" / "part"

    count = tool.write_part(folder, {"a.png": [(0, 0), (256, 0)]}, by_name, 256)

    assert count == 2
    assert sorted(path.name for path in (folder / "images").iterdir()) == ["a_x00000_y00000.png", "a_x00256_y00000.png"]
    assert Image.open(folder / "images" / "a_x00256_y00000.png").size == (256, 256)
    assert Image.open(folder / "masks" / "a_x00000_y00000.png").getpixel((0, 0)) == (0, 0, 255)
    with open(folder / "tile_manifest.csv", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["x"] for row in rows] == ["0", "256"]
    assert tool.part_is_complete(folder, 2)
    assert not tool.part_is_complete(folder, 3)


class _Api(BaseHTTPRequestHandler):
    calls: ClassVar[list[tuple[str, str, bytes]]] = []

    def _reply(self, payload: object, code: int = 200) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        _Api.calls.append(("GET", self.path, b""))
        if self.path == "/api/datasets/build/local-folders" or self.path == "/api/datasets":
            self._reply([])
        elif self.path.startswith("/api/datasets/build/job"):
            index = self.path.rsplit("job", 1)[1]
            record = {"id": int(index), "source": f"data_user/part{index}.csv", "name": f"part {index}"}
            self._reply({"status": "done", "percent": 100, "message": "Done", "dataset": record})
        else:
            self._reply({"error": "nope"}, 404)

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        _Api.calls.append(("POST", self.path, body))
        if self.path == "/api/datasets/build":
            folder = body.decode().split("folder=")[1].split("&")[0]
            self._reply({"job_id": f"job{folder[-1]}"}, 202)
        elif self.path.endswith("/parents"):
            self._reply({"parent_sources": json.loads(body)["parent_sources"]})
        else:
            self._reply({"error": "nope"}, 404)

    def log_message(self, format: str, *args: object) -> None:
        return


@pytest.fixture
def api_url() -> Iterator[str]:
    _Api.calls = []
    server = HTTPServer(("127.0.0.1", 0), _Api)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def _run(data_root: Path, tmp_path: Path, *extra: str) -> int:
    return tool.main(
        [
            "--source", "data_user/src.csv",
            "--data-root", str(data_root),
            "--image-sets", str(tmp_path / "image_sets"),
            "--name", "Demo",
            "--parts", "2",
            *extra,
        ]
    )  # fmt: skip


def test_dry_run_prints_the_plan_and_writes_nothing(
    data_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(data_root, tmp_path, "--dry-run") == 0

    out = capsys.readouterr().out
    assert "2 images (2 with masks) -> 4 tiles of 256x256" in out
    assert "part 1: 1 images, 2 tiles" in out
    assert not (tmp_path / "image_sets").exists()


def test_full_run_tiles_registers_and_links_parts_to_the_source(data_root: Path, tmp_path: Path, api_url: str) -> None:
    assert _run(data_root, tmp_path, "--api", api_url) == 0

    assert (tmp_path / "image_sets" / "demo-part1" / "images").is_dir()
    posts = [(path, body) for method, path, body in _Api.calls if method == "POST"]
    builds = [body.decode() for path, body in posts if path == "/api/datasets/build"]
    assert len(builds) == 2
    assert "source_type=local" in builds[0]
    parents = [json.loads(body) for path, body in posts if path.endswith("/parents")]
    assert parents == [{"parent_sources": ["data_user/src.csv"]}] * 2


def test_no_register_only_writes_tiles(data_root: Path, tmp_path: Path) -> None:
    assert _run(data_root, tmp_path, "--no-register") == 0

    assert (tmp_path / "image_sets" / "demo-part2" / "tile_manifest.csv").is_file()


def test_rerun_skips_parts_that_are_already_tiled(
    data_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _run(data_root, tmp_path, "--no-register")
    capsys.readouterr()

    assert _run(data_root, tmp_path, "--no-register") == 0

    assert capsys.readouterr().out.count("already tiled, skipping") == 2


def test_missing_source_and_bad_arguments_fail_cleanly(
    data_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert tool.main(["--source", "data_user/missing.csv", "--data-root", str(data_root)]) == 1
    assert "not found" in capsys.readouterr().err
    assert _run(data_root, tmp_path, "--tile", "4") == 2


def test_unreachable_api_reports_an_error(
    data_root: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], closed_port_url: str
) -> None:
    assert _run(data_root, tmp_path, "--api", closed_port_url) == 1

    assert "cannot reach the API" in capsys.readouterr().err


def _guide_sources() -> str:
    guide = Path(__file__).resolve().parents[1] / "dashboard" / "src" / "components" / "guide"
    return (guide / "commands.ts").read_text() + (guide / "chapters.tsx").read_text()


def test_every_partition_flag_in_the_guide_exists_in_the_tool() -> None:
    known = set(tool.build_parser()._option_string_actions)
    text = _guide_sources()
    documented = {
        flag
        for line in text.splitlines()
        if "partition_dataset" in line
        for flag in re.findall(r"--[a-z][a-z-]*", line)
    }
    documented |= set(re.findall(r"<code>(--[a-z][a-z-]*)</code>", text))

    assert documented
    assert documented - {"--skip-nima", "--input", "--output"} <= known


def test_the_guide_documents_every_option_of_the_tool() -> None:
    text = _guide_sources()

    for flag in tool.build_parser()._option_string_actions:
        if flag.startswith("--") and flag != "--help":
            assert flag in text, f"{flag} is not documented in the guide"
