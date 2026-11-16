# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

import base64
import json
import socket
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from typing import Any

import httpx2
import pytest
import uvicorn
from mcp import Client
from mcp.client.stdio import StdioServerParameters
from mcp.types import CallToolResult, ImageContent, TextContent

from precisionai.magnidata.mcp_server.client import DEFAULT_API_URL, MagniDataClient
from precisionai.magnidata.mcp_server.conftest import FAKE_URL, JPEG, PNG, SOURCE, FakeAPI
from precisionai.magnidata.mcp_server.server import (
    API_URL_ENV,
    build_parser,
    create_http_app,
    create_server,
    main,
    transport_security,
)

pytestmark = pytest.mark.anyio

TOOLS = {"list_datasets", "describe_dataset", "query_rows", "embedding_projection", "embedding_clusters", "get_image"}


async def call(client: MagniDataClient, tool: str, arguments: dict[str, Any] | None = None) -> CallToolResult:
    async with Client(create_server(client)) as session:
        return await session.call_tool(tool, arguments or {})


async def call_ok(client: MagniDataClient, tool: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    result = await call(client, tool, arguments)
    assert not result.is_error, result.content
    assert result.structured_content is not None
    return result.structured_content


def error_text(result: CallToolResult) -> str:
    assert result.is_error
    content = result.content[0]
    assert isinstance(content, TextContent)
    return content.text


async def test_every_tool_is_listed_as_read_only_with_a_title(client: MagniDataClient):
    async with Client(create_server(client)) as session:
        tools = (await session.list_tools()).tools
    assert {tool.name for tool in tools} == TOOLS
    for tool in tools:
        assert tool.title
        assert tool.description
        assert tool.annotations is not None
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.destructive_hint is False


async def test_list_datasets(client: MagniDataClient):
    result = await call_ok(client, "list_datasets")
    assert result["api_url"] == FAKE_URL
    assert result["datasets"] == [
        {
            "name": "Tiny",
            "source": SOURCE,
            "description": "Four images",
            "row_count": 4,
            "has_embeddings": True,
            "parent_sources": [],
        }
    ]


async def test_describe_dataset(client: MagniDataClient):
    result = await call_ok(client, "describe_dataset", {"source": SOURCE})
    assert result["row_count"] == 4
    assert result["column_count"] == 5
    assert result["image_column"] == "image_path"
    assert result["embedding_variants"] == ["clip"]
    assert result["mask_coverage"] == {"checked": 4, "with_masks": 3}
    assert result["sample_images"] == ["data_user/tiny/a.jpg"]
    columns = {column["name"]: column for column in result["columns"]}
    assert columns["width"]["kind"] == "numeric"
    assert columns["width"]["max"] == 1024
    assert columns["note"]["top_values"][0] == {"value": "sharp", "count": 2}


async def test_describe_dataset_skips_variants_without_embeddings(client: MagniDataClient, fake_api: FakeAPI):
    summary = fake_api.routes["/api/datasets/summary"]

    def without_embeddings(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(summary(request).content) | {"has_embeddings": False, "image_column": None}
        body.pop("mask_coverage")
        return httpx2.Response(200, json=body)

    fake_api.routes["/api/datasets/summary"] = without_embeddings
    result = await call_ok(client, "describe_dataset", {"source": SOURCE})
    assert result["embedding_variants"] == []
    assert result["mask_coverage"] is None
    assert result["image_column"] == "image_path"
    assert all(r.url.path != "/api/dataset/embeddings/variants" for r in fake_api.requests)


async def test_query_rows_filters_sorts_and_pages(client: MagniDataClient):
    result = await call_ok(
        client,
        "query_rows",
        {
            "source": SOURCE,
            "columns": ["image_path", "blur"],
            "filters": [{"column": "width", "op": "<=", "value": 640}],
            "sort_by": "blur",
            "descending": True,
            "limit": 2,
            "offset": 1,
        },
    )
    assert result["total_rows"] == 4
    assert result["matching_rows"] == 3
    assert result["offset"] == 1
    assert result["columns"] == ["image_path", "blur"]
    assert result["rows"] == [
        {"row_index": 1, "image_path": "data_user/tiny/b.jpg", "blur": 3.25},
        {"row_index": 2, "image_path": "data_user/tiny/c.jpg", "blur": None},
    ]


async def test_query_rows_returns_every_column_by_default(client: MagniDataClient):
    result = await call_ok(client, "query_rows", {"source": SOURCE})
    assert result["columns"] == ["image_path", "width", "blur", "category", "note"]
    assert len(result["rows"]) == 4


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"columns": ["nope"]}, "unknown column 'nope'"),
        ({"sort_by": "nope"}, "unknown column 'nope'"),
        ({"filters": [{"column": "nope", "op": "empty"}]}, "unknown column 'nope'"),
        ({"source": "data_user/missing.csv"}, "HTTP 404: dataset not found"),
    ],
)
async def test_query_rows_reports_bad_arguments(client: MagniDataClient, arguments: dict[str, Any], message: str):
    result = await call(client, "query_rows", {"source": SOURCE} | arguments)
    assert message in error_text(result)


@pytest.mark.parametrize("arguments", [{"limit": 0}, {"limit": 201}, {"offset": -1}])
async def test_query_rows_validates_paging(client: MagniDataClient, arguments: dict[str, Any]):
    result = await call(client, "query_rows", {"source": SOURCE} | arguments)
    assert result.is_error


async def test_embedding_projection_pages_points_and_attaches_columns(client: MagniDataClient, fake_api: FakeAPI):
    result = await call_ok(
        client,
        "embedding_projection",
        {"source": SOURCE, "method": "tsne", "include_columns": ["note"], "limit": 2},
    )
    assert fake_api.requests[0].url.params["method"] == "tsne"
    assert result["total_points"] == 4
    assert result["points"] == [
        {"row_index": 0, "x": 0.1235, "y": 1.0, "z": 2.0, "values": {"note": "sharp"}},
        {"row_index": 1, "x": 3.0, "y": 4.0, "z": 5.0, "values": {"note": None}},
    ]


async def test_embedding_projection_without_columns_does_not_fetch_the_table(
    client: MagniDataClient, fake_api: FakeAPI
):
    result = await call_ok(client, "embedding_projection", {"source": SOURCE, "offset": 3, "variant": "clip"})
    assert [p["row_index"] for p in result["points"]] == [3]
    assert result["points"][0]["values"] is None
    assert result["variant"] == "clip"
    assert [r.url.path for r in fake_api.requests] == ["/api/embedding/projection"]


async def test_embedding_projection_rejects_unknown_method(client: MagniDataClient):
    result = await call(client, "embedding_projection", {"source": SOURCE, "method": "umap"})
    assert result.is_error


async def test_embedding_clusters_summarises_each_cluster(client: MagniDataClient, fake_api: FakeAPI):
    result = await call_ok(client, "embedding_clusters", {"source": SOURCE, "k": 2, "examples_per_cluster": 2})
    assert fake_api.requests[0].url.params["k"] == "2"
    assert result["total_points"] == 4
    assert result["labels"] == [1, 0, 1, 1]
    assert result["clusters"] == [
        {
            "label": 1,
            "size": 3,
            "example_rows": [0, 2],
            "example_images": ["data_user/tiny/a.jpg", "data_user/tiny/c.jpg"],
        },
        {"label": 0, "size": 1, "example_rows": [1], "example_images": ["data_user/tiny/b.jpg"]},
    ]


async def test_embedding_clusters_without_examples_skips_the_table(client: MagniDataClient, fake_api: FakeAPI):
    result = await call_ok(
        client, "embedding_clusters", {"source": SOURCE, "examples_per_cluster": 0, "offset": 2, "limit": 1}
    )
    assert result["labels"] == [1]
    assert result["clusters"][0]["example_rows"] == []
    assert [r.url.path for r in fake_api.requests] == ["/api/embedding/clusters"]


@pytest.mark.parametrize(
    ("arguments", "mime_type", "data"),
    [
        ({}, "image/jpeg", JPEG),
        ({"kind": "mask"}, "image/png", PNG),
        ({"kind": "overlay", "alpha": 0.8, "max_size": 1024}, "image/jpeg", JPEG),
    ],
)
async def test_get_image_returns_image_content(
    client: MagniDataClient, fake_api: FakeAPI, arguments: dict[str, Any], mime_type: str, data: bytes
):
    result = await call(client, "get_image", {"path": "data_user/tiny/a.jpg"} | arguments)
    assert not result.is_error
    content = result.content[0]
    assert isinstance(content, ImageContent)
    assert content.mime_type == mime_type
    assert base64.b64decode(content.data) == data
    assert fake_api.last_params()["max_size"] == str(arguments.get("max_size", 512))


async def test_get_image_reports_api_refusal(client: MagniDataClient, fake_api: FakeAPI):
    fake_api.routes["/api/image"] = lambda _: httpx2.Response(400, json={"error": "invalid path"})
    result = await call(client, "get_image", {"path": "/etc/passwd"})
    assert "HTTP 400: invalid path" in error_text(result)


async def test_unreachable_api_is_a_tool_error_not_a_crash():
    unreachable = MagniDataClient("http://127.0.0.1:9", timeout=5)
    result = await call(unreachable, "list_datasets")
    assert "Cannot reach the MagniData API at http://127.0.0.1:9" in error_text(result)


def test_parser_defaults_to_local_stdio(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(API_URL_ENV, raising=False)
    args = build_parser().parse_args([])
    assert args.api_url == DEFAULT_API_URL
    assert args.transport == "stdio"
    assert (args.host, args.port, args.path) == ("127.0.0.1", 8000, "/mcp")


def test_parser_reads_the_api_url_from_the_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv(API_URL_ENV, "https://example.org")
    assert build_parser().parse_args([]).api_url == "https://example.org"
    assert build_parser().parse_args(["--api-url", "http://other:1"]).api_url == "http://other:1"


def test_main_rejects_an_invalid_api_url(capsys: pytest.CaptureFixture[str]):
    with pytest.raises(SystemExit, match="2"):
        main(["--api-url", "localhost:5051"])
    assert "must be an http:// or https:// URL" in capsys.readouterr().err


def test_transport_security_accepts_local_and_listed_hosts_and_origins():
    settings = transport_security(["example.hf.space"], ["https://claude.ai"])
    assert settings.enable_dns_rebinding_protection
    assert {"localhost:*", "127.0.0.1", "example.hf.space"} <= set(settings.allowed_hosts)
    assert {"http://localhost:*", "https://claude.ai"} <= set(settings.allowed_origins)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_port(port: int, process: subprocess.Popen[bytes] | None = None) -> None:
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"server exited with {process.returncode}")
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.05)
    raise TimeoutError(f"nothing listening on port {port}")


@pytest.fixture
def http_server(client: MagniDataClient) -> Iterator[str]:
    port = _free_port()
    app = create_http_app(client, allowed_origins=["https://claude.ai"])
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    _wait_for_port(port)
    yield f"http://127.0.0.1:{port}/mcp"
    server.should_exit = True
    thread.join(timeout=10)


async def test_http_app_serves_the_tools(http_server: str):
    async with Client(http_server) as session:
        tools = (await session.list_tools()).tools
        result = await session.call_tool("list_datasets", {})
    assert {tool.name for tool in tools} == TOOLS
    assert result.structured_content is not None
    assert result.structured_content["datasets"][0]["source"] == SOURCE


@pytest.mark.parametrize(
    ("origin", "status"),
    [("https://evil.example", 403), ("http://localhost:3000", 200), ("https://claude.ai", 200), (None, 200)],
)
async def test_http_app_validates_the_origin_header(http_server: str, origin: str | None, status: int):
    headers = {"accept": "application/json, text/event-stream", "content-type": "application/json"}
    if origin:
        headers["origin"] = origin
    request = {"jsonrpc": "2.0", "id": 1, "method": "ping"}
    async with httpx2.AsyncClient() as http:
        response = await http.post(http_server, json=request, headers=headers)
    assert response.status_code == status


async def test_stdio_command_starts_and_answers():
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "precisionai.magnidata.mcp_server", "--api-url", "http://127.0.0.1:9", "--timeout", "5"],
    )
    async with Client(parameters) as session:
        tools = (await session.list_tools()).tools
        result = await session.call_tool("list_datasets", {})
    assert {tool.name for tool in tools} == TOOLS
    assert "Cannot reach the MagniData API" in error_text(result)


async def test_streamable_http_command_serves_on_the_given_port():
    port = _free_port()
    command = [sys.executable, "-m", "precisionai.magnidata.mcp_server", "--transport", "streamable-http"]
    process = subprocess.Popen([*command, "--port", str(port), "--path", "/tools"], stderr=subprocess.DEVNULL)
    try:
        _wait_for_port(port, process)
        async with Client(f"http://127.0.0.1:{port}/tools") as session:
            tools = (await session.list_tools()).tools
        assert {tool.name for tool in tools} == TOOLS
    finally:
        process.terminate()
        process.wait(timeout=10)
