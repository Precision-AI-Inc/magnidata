# Copyright 2026 Precision AI
# SPDX-License-Identifier: Apache-2.0

import re

import httpx2
import pytest

from precisionai.magnidata.mcp_server.client import (
    DEMO_API_URL,
    MagniDataAPIError,
    MagniDataClient,
    normalize_api_url,
)
from precisionai.magnidata.mcp_server.conftest import FAKE_URL, JPEG, LABELS, PNG, POSITIONS, SOURCE, TINY_CSV, FakeAPI
from precisionai.magnidata.mcp_server.schemas import ImageKind

pytestmark = pytest.mark.anyio


def _client_answering(handler: httpx2.MockTransport) -> MagniDataClient:
    return MagniDataClient(FAKE_URL, transport=handler)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://localhost:5051", "http://localhost:5051"),
        ("http://localhost:5051/", "http://localhost:5051"),
        (" https://example.org/api/ ", "https://example.org"),
        ("https://example.org/sub/api", "https://example.org/sub"),
    ],
)
async def test_normalize_api_url(url: str, expected: str):
    assert normalize_api_url(url) == expected


@pytest.mark.parametrize("url", ["localhost:5051", "ftp://example.org", "http://", ""])
async def test_normalize_api_url_rejects_non_http_urls(url: str):
    with pytest.raises(ValueError, match="must be an http:// or https:// URL"):
        normalize_api_url(url)


async def test_read_endpoints_send_the_expected_requests(client: MagniDataClient, fake_api: FakeAPI):
    assert (await client.list_datasets())[0]["source"] == SOURCE
    assert (await client.dataset_summary(SOURCE))["image_column"] == "image_path"
    assert fake_api.last_params() == {"source": SOURCE}
    assert await client.dataset_csv(SOURCE) == TINY_CSV
    assert await client.embedding_projection(SOURCE, "tsne", "clip") == POSITIONS
    assert fake_api.last_params() == {"source": SOURCE, "method": "tsne", "variant": "clip"}
    assert await client.embedding_clusters(SOURCE, 5) == LABELS
    assert fake_api.last_params() == {"source": SOURCE, "k": "5"}
    assert fake_api.requests[0].headers["user-agent"].startswith("magnidata-mcp/")


async def test_embedding_variants_leave_out_the_projections_sidecar(client: MagniDataClient):
    assert await client.embedding_variants(SOURCE) == ["clip"]


@pytest.mark.parametrize(
    ("kind", "route", "media_type", "data"),
    [
        ("image", "/api/image", "image/jpeg", JPEG),
        ("mask", "/api/annotation/mask", "image/png", PNG),
        ("overlay", "/api/annotation/overlay", "image/jpeg", JPEG),
    ],
)
async def test_image_kinds_map_to_their_routes(
    client: MagniDataClient, fake_api: FakeAPI, kind: ImageKind, route: str, media_type: str, data: bytes
):
    assert await client.image("data_user/tiny/a.jpg", kind, 256, 0.3) == (data, media_type)
    assert fake_api.requests[-1].url.path == route
    expected = {"path": "data_user/tiny/a.jpg", "max_size": "256"} | ({"alpha": "0.3"} if kind == "overlay" else {})
    assert fake_api.last_params() == expected


async def test_image_rejects_a_response_that_is_not_an_image(fake_api: FakeAPI):
    fake_api.routes["/api/image"] = lambda _: httpx2.Response(200, text="<html></html>")
    client = _client_answering(httpx2.MockTransport(fake_api))
    with pytest.raises(MagniDataAPIError, match="returned text/plain, not an image"):
        await client.image("a.jpg", "image", 256, 0.5)


async def test_api_error_message_is_passed_on(client: MagniDataClient):
    with pytest.raises(MagniDataAPIError, match="HTTP 404: dataset not found"):
        await client.dataset_csv("data_user/missing.csv")


async def test_html_error_page_is_reduced_to_the_status(fake_api: FakeAPI):
    fake_api.routes["/api/embedding/projection"] = lambda _: httpx2.Response(500, text="<!doctype html><p>boom</p>")
    client = _client_answering(httpx2.MockTransport(fake_api))
    with pytest.raises(MagniDataAPIError, match=r"HTTP 500: Internal Server Error$"):
        await client.embedding_projection(SOURCE, "pca")


async def test_json_error_without_message_is_quoted(fake_api: FakeAPI):
    fake_api.routes["/api/datasets"] = lambda _: httpx2.Response(503, json={"status": "starting"})
    client = _client_answering(httpx2.MockTransport(fake_api))
    with pytest.raises(MagniDataAPIError, match=re.escape("HTTP 503: {'status': 'starting'}")):
        await client.list_datasets()


async def test_non_json_answer_is_reported(fake_api: FakeAPI):
    fake_api.routes["/api/datasets"] = lambda _: httpx2.Response(200, text="<html>dashboard</html>")
    client = _client_answering(httpx2.MockTransport(fake_api))
    with pytest.raises(MagniDataAPIError, match="did not return JSON"):
        await client.list_datasets()


async def test_dataset_list_must_be_a_list(fake_api: FakeAPI):
    fake_api.routes["/api/datasets"] = lambda _: httpx2.Response(200, json={"datasets": []})
    client = _client_answering(httpx2.MockTransport(fake_api))
    with pytest.raises(MagniDataAPIError, match="did not return a list"):
        await client.list_datasets()


async def test_unreachable_api_suggests_the_stack_or_the_demo():
    def refuse(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    client = _client_answering(httpx2.MockTransport(refuse))
    with pytest.raises(MagniDataAPIError, match=re.escape(f"Cannot reach the MagniData API at {FAKE_URL}")) as info:
        await client.list_datasets()
    assert DEMO_API_URL in str(info.value)


async def test_slow_api_reports_the_timeout():
    def stall(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("timed out", request=request)

    client = MagniDataClient(FAKE_URL, timeout=2.5, transport=httpx2.MockTransport(stall))
    with pytest.raises(MagniDataAPIError, match=r"did not answer GET /api/datasets within 2\.5 s"):
        await client.list_datasets()
